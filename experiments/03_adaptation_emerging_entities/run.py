from collections import defaultdict
import copy
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import shutil
from elasticsearch import Elasticsearch
from typing import List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
import json
from lightning.fabric.utilities.seed import seed_everything
from haystack import Pipeline
from haystack.nodes import JoinDocuments
import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_json, tokenize_search_results_json, query_database
from models.asp_t5 import ASPT5Model, get_tokenizer
from pipelines.evaluation import factors
from configs.asp_t5 import BEST_WNUT_T5_ASP, WNUT_T5_ASP_LOWNERGAZ_SENT, BEST_WNUT_T5_ASP_LOWNERGAZ_SENT, WORST_WNUT_T5_ASP_LOWNERGAZ_SENT, T5_ASP_LOWNERGAZ_SENT, WNUT_T5_ASP
from hyperparameter_tuning.utils import get_search_results, get_search_results_filtered
from data_augmentation.sampling import per_type_uniform_sampling
from search.lownergaz.setup import add_lownergaz_search_components
from search.sent.setup import add_sent_search_components
from search.utils import get_gazetteers_from_documents
from data_augmentation.augments import make_erroneous_dataset

from itertools import product

files = {
    "types": os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    "wnut_train": os.path.join(thesis_path, "data", "wnut", "wnut_train.json"),
    "wnut_dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
    "wnut_test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
}

seeds = [1, 2, 3]

elasticsearch_client = Elasticsearch("http://localhost:9200")

configs = {
    ("full", "vanilla"): WNUT_T5_ASP,
    ("full", "vanilla-pretrained"): BEST_WNUT_T5_ASP,
    ("no", "vanilla-pretrained"): BEST_WNUT_T5_ASP,
    ("full", "no-pretrained"): WNUT_T5_ASP_LOWNERGAZ_SENT,
    ("full", "best-pretrained"): BEST_WNUT_T5_ASP_LOWNERGAZ_SENT,
    ("full", "worst-pretrained"): WORST_WNUT_T5_ASP_LOWNERGAZ_SENT,
    ("no", "best-pretrained"): T5_ASP_LOWNERGAZ_SENT,
    ("no", "worst-pretrained"): T5_ASP_LOWNERGAZ_SENT,
}

for config in configs.values():
    if config is not None:
        config.update({
            "data_path":
            os.path.join(thesis_path, "experiments",
                         "03_adaptation_emerging_entities", "data")
        })

database_combinations = [
    ("wnut_train", ),
    ("wnut_train", "wnut_dev"),
    ("wnut_train", "wnut_dev", "wnut_test"),
    ("lownergaz_sent", ),
    ("lownergaz_sent", "wnut_train"),
    ("lownergaz_sent", "wnut_train", "wnut_dev"),
    ("lownergaz_sent", "wnut_train", "wnut_dev", "wnut_test"),
]

with open(files["wnut_train"], encoding="utf-8") as file:
    wnut_train = json.load(file)

with open(files["wnut_dev"], encoding="utf-8") as file:
    wnut_dev = json.load(file)

with open(files["wnut_test"], encoding="utf-8") as file:
    wnut_test = json.load(file)

with open(files["types"], encoding="utf-8") as file:
    types = json.load(file)

datasets = {
    "wnut_train": wnut_train,
    "wnut_dev": wnut_dev,
    "wnut_test": wnut_test,
    "types": types
}

total = product(seeds, list(configs.keys()))


def get_validation_dataloader(config, dataset: Dataset):
    return DataLoader(dataset,
                      batch_size=int(config["batch_size"] * 4),
                      collate_fn=ner_collate_fn,
                      num_workers=3,
                      persistent_workers=False,
                      pin_memory=True,
                      shuffle=False,
                      prefetch_factor=20)


def train_model(seed: int, config: dict, tokenized_files: dict,
                ckpt_path: Optional[str]):

    train_config = copy.deepcopy(config)
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    seed_everything(seed)

    grad_accum_steps = factors(config["batch_size"])
    tokenizer = get_tokenizer(config)

    processor = NERDataProcessor(train_config,
                                 tokenizer,
                                 tokenized_files["train"],
                                 tokenized_files["dev"],
                                 None,
                                 files["types"],
                                 use_cache=False)
    config["num_labels"] = len(processor.labels)

    train, val, _ = processor.get_tensor_samples()
    train_config["train_len"] = len(train)

    collator = ner_collate_fn
    if "train_search_dropout" in config:
        collator = NERCollator(config["train_search_dropout"], False)

    train_config["fused"] = True
    train_config["precision"] = "bf16-mixed"
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

    # Checkpoints
    checkpoint_best = ModelCheckpoint(dirpath=checkpoint_base_path,
                                      filename="best",
                                      monitor="val_f1",
                                      mode="max",
                                      save_top_k=1)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join([str(seed), f"{finetuning}_{pretrained}"]),
    )

    def get_dataloaders():
        # Train loader
        train_loader = DataLoader(train,
                                  batch_size=train_config["batch_size"],
                                  collate_fn=collator,
                                  num_workers=3,
                                  persistent_workers=False,
                                  pin_memory=True,
                                  shuffle=True,
                                  prefetch_factor=20)
        # Validation loaders
        dev_val_loader = DataLoader(val,
                                    batch_size=int(train_config["batch_size"] *
                                                   4),
                                    collate_fn=ner_collate_fn,
                                    num_workers=3,
                                    persistent_workers=False,
                                    pin_memory=True,
                                    shuffle=False,
                                    prefetch_factor=20)
        return train_loader, dev_val_loader

    def get_model_trainer():
        if ckpt_path is not None:
            model = ASPT5Model.load_from_checkpoint(ckpt_path,
                                                    config=train_config,
                                                    tokenizer=tokenizer)
        else:
            model = ASPT5Model(train_config, tokenizer)
        trainer = pl.Trainer(accelerator="gpu",
                             logger=tb_logger,
                             devices=1,
                             log_every_n_steps=train_config["batch_size"] *
                             train_config["gradient_accumulation_steps"],
                             accumulate_grad_batches=train_config[
                                 "gradient_accumulation_steps"],
                             precision=train_config["precision"],
                             max_epochs=train_config["num_epochs"],
                             check_val_every_n_epoch=1,
                             num_sanity_val_steps=0,
                             enable_checkpointing=True,
                             enable_progress_bar=True,
                             callbacks=[checkpoint_best])
        return model, trainer

    train_loader, dev_val_loader = get_dataloaders()
    model, trainer = get_model_trainer()
    trained = False
    while not trained:
        try:
            trainer.fit(model, train_loader, val_dataloaders=dev_val_loader)
            # save last model
            trainer.save_checkpoint(
                os.path.join(checkpoint_base_path, "last.ckpt"))
            trained = True
        except Exception as e:
            print(e)
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]
            train_loader, dev_val_loader = get_dataloaders()
            model, trainer = get_model_trainer()
    return os.path.join(checkpoint_base_path,
                        "last.ckpt"), os.path.join(checkpoint_base_path,
                                                   "best.ckpt")


def test_model(config, best_ckpt_path, last_ckpt_path, dataset: Dataset, name):
    metrics_base_path = os.path.join(config["data_path"], f"seed_{str(seed)}",
                                     "04_metrics",
                                     f"{finetuning}_{pretrained}",
                                     "_".join(database_comb))
    os.makedirs(metrics_base_path, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         precision="bf16-mixed",
                         num_sanity_val_steps=0,
                         enable_checkpointing=False,
                         enable_progress_bar=True)
    val_loader = get_validation_dataloader(config, dataset)

    def save_metrics(model, dataset, checkpoint):
        with open(
                os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"),
                "wb") as file:
            pickle.dump(model.test_metrics, file)

    # test model
    last_model = ASPT5Model.load_from_checkpoint(last_ckpt_path)
    trainer.test(last_model, val_loader)
    save_metrics(last_model, name, "last")
    best_model = ASPT5Model.load_from_checkpoint(best_ckpt_path)
    trainer.test(best_model, val_loader)
    save_metrics(best_model, name, "best")


def setup_database(sent_search_algorithm: str,
                   sent_search_topk: int,
                   gaz_search_algorithm: str,
                   gaz_search_topk: int,
                   join_method: str,
                   join_topk: int,
                   reset=False,
                   name: str = "lownergaz",
                   sents: List[dict] = [],
                   gazs: List[dict] = []):
    search = Pipeline()
    join_documents_input = []
    # sentences
    sent_name = "sent" if name == "lownergaz" else name + "_sent"
    add_sent_search_components(search, sent_search_algorithm, sent_search_topk,
                               join_documents_input, reset, sent_name, sents)

    # lowner gazetteers
    lownergaz_name = name if name == "lownergaz" else name + "_lownergaz"
    add_lownergaz_search_components(search, gaz_search_algorithm,
                                    gaz_search_topk, join_documents_input,
                                    reset, lownergaz_name, gazs)

    # join documents

    join_documents = JoinDocuments(join_mode=join_method, top_k_join=join_topk)
    search.add_node(join_documents, "DocumentJoin", join_documents_input)

    return search


def get_tokenized_filepath(config, file_path, search_results, data_path):
    tokenizer = get_tokenizer(config)
    use_labels = config["use_labels"] if "use_labels" in config else None
    use_mentions = config["use_mentions"] if "use_mentions" in config else None
    sent_use_labels = config[
        "sent_use_labels"] if "sent_use_labels" in config else None
    sent_use_mentions = config[
        "sent_use_mentions"] if "sent_use_mentions" in config else None
    gaz_use_labels = config[
        "gaz_use_labels"] if "gaz_use_labels" in config else None
    gaz_use_mentions = config[
        "gaz_use_mentions"] if "gaz_use_mentions" in config else None

    return tokenize_search_results_json(tokenizer,
                                        file_path,
                                        files["types"],
                                        search_results,
                                        data_path,
                                        use_labels=use_labels,
                                        use_mentions=use_mentions,
                                        sent_use_labels=sent_use_labels,
                                        sent_use_mentions=sent_use_mentions,
                                        gaz_use_labels=gaz_use_labels,
                                        gaz_use_mentions=gaz_use_mentions,
                                        prepend_search_results=False)


# prepare databases
config = list(configs.values())[0]
for database_comb in database_combinations:
    name = "_".join(database_comb)
    if "lownergaz_sent" in database_comb and len(database_comb) > 1:
        if not elasticsearch_client.indices.exists(
                index=name +
                "_lownergaz") or not elasticsearch_client.indices.exists(
                    index=name + "_sent"):
            dataset = []
            for dataset_name in database_comb[1:]:
                dataset.extend(datasets[dataset_name])
            # copy lownergaz database
            elasticsearch_client.indices.clone(index="lownergaz",
                                               target=name + "_lownergaz")
            elasticsearch_client.indices.put_settings(
                index=name + "_lownergaz",
                body={"index": {
                    "blocks": {
                        "write": False
                    }
                }})

            # copy sent database
            elasticsearch_client.indices.clone(index="sent",
                                               target=name + "_sent")
            elasticsearch_client.indices.put_settings(
                index=name + "_sent",
                body={"index": {
                    "blocks": {
                        "write": False
                    }
                }})
            # populate database
            search = setup_database(config["sent_search_algorithm"],
                                    config["sent_search_topk"],
                                    config["gaz_search_algorithm"],
                                    config["gaz_search_topk"],
                                    config["search_join_method"],
                                    config["search_topk"],
                                    name=name,
                                    gazs=dataset,
                                    sents=dataset)
        elif "lownergaz_sent" not in database_comb:
            if not elasticsearch_client.indices.exists(
                    index=name +
                    "_lownergaz") or not elasticsearch_client.indices.exists(
                        index=name + "_sent"):
                dataset = []
                for dataset_name in database_comb:
                    dataset.extend(datasets[dataset_name])
                # populate database
                search = setup_database(config["sent_search_algorithm"],
                                        config["sent_search_topk"],
                                        config["gaz_search_algorithm"],
                                        config["gaz_search_topk"],
                                        config["search_join_method"],
                                        config["search_topk"],
                                        name=name,
                                        gazs=dataset,
                                        sents=dataset)

tokenized_files = defaultdict(dict)
search_configs = set([
    (config["sent_search_algorithm"], config["sent_search_topk"],
     config["gaz_search_algorithm"], config["gaz_search_topk"],
     config["search_join_method"], config["search_topk"],
     config["data_path"]) if "search_join_method" in config else
    ("None", "None", "None", "None", "None", "None", config["data_path"])
    for config in configs.values() if config is not None
])
for search_config in search_configs:
    sent_search_algorithm, sent_search_topk, gaz_search_algorithm, gaz_search_topk, search_join_method, search_topk, data_path = search_config
    search_config_name = "_".join([
        sent_search_algorithm,
        str(sent_search_topk), gaz_search_algorithm,
        str(gaz_search_topk), search_join_method,
        str(search_topk)
    ])
    if search_join_method == "None":
        tokenized_data_path = os.path.join(config["data_path"],
                                           "02_tokenized_dataset",
                                           search_config_name)
        os.makedirs(tokenized_data_path, exist_ok=True)
        tokenizer = get_tokenizer(config)
        for part in ["train", "dev", "test"]:
            tokenized_files[search_config_name][part] = tokenize_json(
                tokenizer, files[f"wnut_{part}"], files["types"],
                tokenized_data_path)
    else:

        for database_comb in database_combinations:
            # get search results
            search_base_path = os.path.join(config["data_path"],
                                            "01_search_results",
                                            search_config_name,
                                            "_".join(database_comb))
            os.makedirs(search_base_path, exist_ok=True)

            def get_search():
                if len(database_comb) > 1:
                    name = "_".join(database_comb)
                    search = setup_database(sent_search_algorithm,
                                            sent_search_topk,
                                            gaz_search_algorithm,
                                            gaz_search_topk,
                                            search_join_method,
                                            search_topk,
                                            name=name)
                else:
                    search = setup_database(sent_search_algorithm,
                                            sent_search_topk,
                                            gaz_search_algorithm,
                                            gaz_search_topk,
                                            search_join_method, search_topk)
                return search

            train_search_path = os.path.join(search_base_path,
                                             "wnut_train.pkl")
            if not os.path.exists(train_search_path):
                search = get_search()
                search_results_train = get_search_results_filtered(
                    search, datasets["wnut_train"], True)
                with open(train_search_path, "wb") as file:
                    pickle.dump(search_results_train, file)
            else:
                with open(train_search_path, "rb") as file:
                    search_results_train = pickle.load(file)

            dev_search_path = os.path.join(search_base_path, "wnut_dev.pkl")
            if not os.path.exists(dev_search_path):
                search = get_search()
                search_results_dev = get_search_results(
                    search, datasets["wnut_dev"])
                with open(dev_search_path, "wb") as file:
                    pickle.dump(search_results_dev, file)
            else:
                with open(dev_search_path, "rb") as file:
                    search_results_dev = pickle.load(file)

            test_search_path = os.path.join(search_base_path, "wnut_test.pkl")
            if not os.path.exists(test_search_path):
                search = get_search()
                search_results_test = get_search_results(
                    search, datasets["wnut_test"])
                with open(test_search_path, "wb") as file:
                    pickle.dump(search_results_test, file)
            else:
                with open(test_search_path, "rb") as file:
                    search_results_test = pickle.load(file)

            # prep data
            tokenized_data_path = os.path.join(config["data_path"],
                                               "02_tokenized_dataset",
                                               search_config_name,
                                               "_".join(database_comb))
            os.makedirs(tokenized_data_path, exist_ok=True)
            tokenized_files[search_config_name][database_comb] = {}
            for part in ["train", "dev", "test"]:
                wnut_tokenized_path = os.path.join(
                    config["data_path"], "02_tokenized_dataset",
                    search_config_name, "_".join(database_comb),
                    f"wnut_{part}.t5-small.jsonlines")
                if not os.path.exists(wnut_tokenized_path):
                    get_tokenized_filepath(config, files[f"wnut_{part}"],
                                           search_results_train,
                                           tokenized_data_path)
                tokenized_files[search_config_name][database_comb][
                    part] = wnut_tokenized_path

for seed, (finetuning, pretrained) in total:
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    seed_everything(seed)
    config = configs[(finetuning, pretrained)]

    if "search_join_method" in config:
        search_config_name = "_".join([
            config["sent_search_algorithm"],
            str(config["sent_search_topk"]),
            config["gaz_search_algorithm"],
            str(config["gaz_search_topk"]),
            config["search_join_method"],
            str(config["search_topk"]),
        ])
    else:
        search_config_name = "_".join([
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
        ])

    # finetune models -> get last + best ckpt path
    checkpoint_base_path = os.path.join(config["data_path"],
                                        f"seed_{str(seed)}", "03_checkpoints",
                                        f"{finetuning}_{pretrained}")
    os.makedirs(checkpoint_base_path, exist_ok=True)
    if finetuning == "no":
        # copy best + last ckpt
        ckpt_path = BEST_WNUT_T5_ASP["ckpt_path"]
        if pretrained == "best-pretrained":
            ckpt_path = BEST_WNUT_T5_ASP_LOWNERGAZ_SENT["ckpt_path"]
        if pretrained == "worst-pretrained":
            ckpt_path = WORST_WNUT_T5_ASP_LOWNERGAZ_SENT["ckpt_path"]
        ckpt_dir_path = os.path.dirname(
            BEST_WNUT_T5_ASP_LOWNERGAZ_SENT["ckpt_path"])
        for file_name in ["best.ckpt", "last.ckpt"]:
            shutil.copy(os.path.join(ckpt_dir_path, file_name),
                        os.path.join(checkpoint_base_path, file_name))
        last_ckpt_path = os.path.join(checkpoint_base_path, "last.ckpt")
        best_ckpt_path = os.path.join(checkpoint_base_path, "best.ckpt")
    elif finetuning == "full":
        # take best / worst ckpt / None for finetuning
        ckpt_path = None
        if pretrained == "best-pretrained":
            ckpt_path = BEST_WNUT_T5_ASP_LOWNERGAZ_SENT["ckpt_path"]
        elif pretrained == "worst-pretrained":
            ckpt_path = WORST_WNUT_T5_ASP_LOWNERGAZ_SENT["ckpt_path"]
        elif pretrained == "vanilla-pretrained":
            ckpt_path = BEST_WNUT_T5_ASP["ckpt_path"]

        relevant_files = tokenized_files[search_config_name]
        if "search_join_method" in config:
            relevant_files = tokenized_files[search_config_name][
                database_combinations[0]]

        last_ckpt_path, best_ckpt_path = train_model(seed, config,
                                                     relevant_files, ckpt_path)
    else:
        last_ckpt_path = ""
        best_ckpt_path = ""

    # test models with augmented datasets
    if "vanilla" in pretrained:
        processor = NERDataProcessor(
            config,
            get_tokenizer(config),
            tokenized_files[search_config_name]["train"],
            tokenized_files[search_config_name]["dev"],
            tokenized_files[search_config_name]["test"],
            files["types"],
            use_cache=False)
        train_dataset, dev_dataset, test_dataset = processor.get_tensor_samples(
        )
        test_model(config, best_ckpt_path, last_ckpt_path, train_dataset,
                   "train")
        test_model(config, best_ckpt_path, last_ckpt_path, dev_dataset, "dev")
        test_model(
            config,
            best_ckpt_path,
            last_ckpt_path,
            test_dataset,  # type: ignore
            "test")
    else:
        for database_comb in database_combinations:
            processor = NERDataProcessor(
                config,
                get_tokenizer(config),
                tokenized_files[search_config_name][database_comb]["train"],
                tokenized_files[search_config_name][database_comb]["dev"],
                tokenized_files[search_config_name][database_comb]["test"],
                files["types"],
                use_cache=False)
            train_dataset, dev_dataset, test_dataset = processor.get_tensor_samples(
            )
            test_model(config, best_ckpt_path, last_ckpt_path, train_dataset,
                       "train")
            test_model(config, best_ckpt_path, last_ckpt_path, dev_dataset,
                       "dev")
            test_model(
                config,
                best_ckpt_path,
                last_ckpt_path,
                test_dataset,  # type: ignore
                "test")
