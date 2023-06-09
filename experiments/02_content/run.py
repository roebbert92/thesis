import copy
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

from typing import List
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
from data_preprocessing.tokenize import tokenize_search_results_json, query_database
from models.asp_t5 import ASPT5Model, get_tokenizer
from pipelines.evaluation import factors
from configs.asp_t5 import T5_ASP_LOWNERGAZ_SENT
from hyperparameter_tuning.utils import get_search_results
from data_augmentation.sampling import per_type_uniform_sampling
from search.lownergaz.setup import add_lownergaz_search_components
from search.sent.setup import add_sent_search_components
from search.utils import get_gazetteers_from_documents
from data_augmentation.augments import make_erroneous_dataset

files = {
    "types":
    os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    "train":
    os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
    "dev":
    os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
    "test":
    os.path.join(
        thesis_path,
        "data",
        "mlowner",
        #"lowner_test.json"
        "lowner_dev.json"),
    "multiconer":
    os.path.join(thesis_path, "data", "multiconer", "multiconer_test.json"),
    "lownergaz":
    os.path.join(thesis_path, "data", "mlowner", "lowner_gazetteer.json"),
}

with open(files["train"], encoding="utf-8") as file:
    lowner_train = json.load(file)

with open(files["dev"], encoding="utf-8") as file:
    lowner_dev = json.load(file)

with open(files["test"], encoding="utf-8") as file:
    lowner_test = json.load(file)

with open(files["multiconer"], encoding="utf-8") as file:
    multiconer = json.load(file)

with open(files["types"], encoding="utf-8") as file:
    types = json.load(file)

seeds = [1, 2, 3]
datasets = {
    "train": lowner_train,
    "dev": lowner_dev,
    "test": lowner_test,
    "multiconer": multiconer,
    "types": types
}
gazetteer_sizes = [2000, 4000, 6000]
error_percent_ratios = [5, 10, 15]

config = T5_ASP_LOWNERGAZ_SENT
config.update({
    "data_path":
    os.path.join(thesis_path, "experiments", "02_content", "data")
})


def get_validation_dataloader(config, dataset: Dataset):
    return DataLoader(dataset,
                      batch_size=int(config["batch_size"] * 4),
                      collate_fn=ner_collate_fn,
                      num_workers=3,
                      persistent_workers=False,
                      pin_memory=True,
                      shuffle=False,
                      prefetch_factor=20)


def train_model(seed: int, gazetteer_size: int, error_percent_ratio: int,
                config, train: Dataset, val: Dataset):

    grad_accum_steps = factors(config["batch_size"])
    tokenizer = get_tokenizer(config)

    collator = NERCollator(config["train_search_dropout"], False)

    config["fused"] = True
    config["precision"] = "bf16-mixed"
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    train_config = copy.deepcopy(config)
    trained = False

    # Checkpoints
    checkpoint_base_path = os.path.join(config["data_path"],
                                        f"seed_{str(seed)}", "03_checkpoints",
                                        f"size_{gazetteer_size}",
                                        f"error_ratio_{error_percent_ratio}")
    checkpoint_best = ModelCheckpoint(dirpath=checkpoint_base_path,
                                      filename="best",
                                      monitor="val_f1",
                                      mode="max",
                                      save_top_k=1)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join([
            str(seed), f"size_{gazetteer_size}",
            f"error_ratio_{error_percent_ratio}"
        ]),
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
        dev_val_loader = get_validation_dataloader(train_config, val)
        return train_loader, dev_val_loader

    def get_model_trainer():
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
                                     "04_metrics", f"size_{gazetteer_size}",
                                     f"error_ratio_{error_percent_ratio}")
    os.makedirs(metrics_base_path, exist_ok=True)
    tokenizer = get_tokenizer(config)
    model = ASPT5Model(config, tokenizer)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         precision=config["precision"],
                         num_sanity_val_steps=0,
                         enable_checkpointing=False,
                         enable_progress_bar=True)
    val_loader = get_validation_dataloader(config, dataset)

    def save_metrics(dataset, checkpoint):
        with open(
                os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"),
                "wb") as file:
            pickle.dump(model.test_metrics, file)

    # test model
    trainer.test(model, val_loader, ckpt_path=last_ckpt_path)
    save_metrics(name, "last")
    trainer.test(model, val_loader, ckpt_path=best_ckpt_path)
    save_metrics(name, "best")


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


# Datapoints
## 00_datasets
## 01_search_results
## 02_tokenized_datasets
## 03_checkpoints
## 04_metrics

for seed in seeds:
    # seed
    seed_everything(seed)
    for gazetteer_size in gazetteer_sizes:
        for error_percent_ratio in error_percent_ratios:
            error_ratio = error_percent_ratio / 100
            search_results = {}
            # create gazetteer split of multiconer + get top-5 similar gazetteers of lownergaz --> save in 00_datasets
            sampled_multiconer, _ = per_type_uniform_sampling(
                datasets["multiconer"], list(datasets["types"]),
                gazetteer_size)

            lowner_gaz_search = Pipeline()
            add_lownergaz_search_components(lowner_gaz_search,
                                            config["gaz_search_algorithm"], 5)
            sampled_multiconer_gaz = [{
                "extended": str(doc.content).split(" ")
            } for doc in get_gazetteers_from_documents(sampled_multiconer)]
            sampled_lownergaz = []
            sampled_lownergaz_ids = set()
            for _, results in query_database(sampled_multiconer_gaz,
                                             lowner_gaz_search):
                for result in results:
                    if result.id not in sampled_lownergaz_ids:
                        sampled_lownergaz_ids.add(result.id)
                        tokens = str(result.content).split(" ")
                        sampled_lownergaz.append({
                            "tokens":
                            tokens,
                            "entities": [{
                                "start": 0,
                                "end": len(tokens),
                                "type": result.meta["type"],
                            }],
                            "doc_id":
                            result.id
                        })

            # save clean mulitconer + lownergaz gazetteer splits
            dataset_base_path = os.path.join(
                config["data_path"], f"seed_{seed}", "00_datasets",
                f"size_{gazetteer_size}", f"error_ratio_{error_percent_ratio}")
            os.makedirs(dataset_base_path, exist_ok=True)
            with open(os.path.join(dataset_base_path,
                                   "clean_multiconer_split.json"),
                      "w",
                      encoding="utf-8") as file:
                json.dump(sampled_multiconer, file)
            with open(os.path.join(dataset_base_path,
                                   "clean_lownergaz_split.json"),
                      "w",
                      encoding="utf-8") as file:
                json.dump(sampled_lownergaz, file)

            # create augmented lowner train, dev + gazetteer split
            aug_lowner_train = make_erroneous_dataset(datasets["train"],
                                                      datasets["types"],
                                                      error_ratio)
            aug_lowner_dev = make_erroneous_dataset(datasets["dev"],
                                                    datasets["types"],
                                                    error_ratio)
            aug_multiconer = make_erroneous_dataset(sampled_multiconer,
                                                    datasets["types"],
                                                    error_ratio)
            aug_lownergaz = make_erroneous_dataset(sampled_lownergaz,
                                                   datasets["types"],
                                                   error_ratio)

            # save augmented datasets
            aug_lowner_train_path = os.path.join(
                dataset_base_path, "augmented_lowner_train.json")
            with open(aug_lowner_train_path, "w", encoding="utf-8") as file:
                json.dump(aug_lowner_train, file)
            aug_lowner_dev_path = os.path.join(dataset_base_path,
                                               "augmented_lowner_dev.json")
            with open(aug_lowner_dev_path, "w", encoding="utf-8") as file:
                json.dump(aug_lowner_dev, file)
            with open(os.path.join(dataset_base_path,
                                   "augmented_multiconer_split.json"),
                      "w",
                      encoding="utf-8") as file:
                json.dump(aug_multiconer, file)
            with open(os.path.join(dataset_base_path,
                                   "augmented_lownergaz_split.json"),
                      "w",
                      encoding="utf-8") as file:
                json.dump(aug_lownergaz, file)

            # setup database with augmented gazetteer split
            augmented_split_search = setup_database(
                config["sent_search_algorithm"],
                config["sent_search_topk"],
                config["gaz_search_algorithm"],
                config["gaz_search_topk"],
                config["search_join_method"],
                config["search_topk"],
                reset=True,
                name=f"{seed}_{gazetteer_size}_{error_percent_ratio}_aug_split",
                sents=aug_multiconer,
                gazs=aug_lownergaz)
            # get search results for augmented lowner train + dev, clean lowner test
            aug_search_results_aug_train = get_search_results(
                augmented_split_search, aug_lowner_train)
            aug_search_results_aug_dev = get_search_results(
                augmented_split_search, aug_lowner_dev)
            aug_search_results_test = get_search_results(
                augmented_split_search, datasets["test"])

            # save augmented search results
            search_base_path = os.path.join(
                config["data_path"], f"seed_{str(seed)}", "01_search_results",
                f"size_{gazetteer_size}", f"error_ratio_{error_percent_ratio}")
            os.makedirs(search_base_path, exist_ok=True)
            with open(
                    os.path.join(search_base_path,
                                 "aug_search_results_aug_train.pkl"),
                    "wb") as file:
                pickle.dump(aug_search_results_aug_train, file)
            with open(
                    os.path.join(search_base_path,
                                 "aug_search_results_aug_dev.pkl"),
                    "wb") as file:
                pickle.dump(aug_search_results_aug_dev, file)
            with open(
                    os.path.join(search_base_path,
                                 "aug_search_results_test.pkl"), "wb") as file:
                pickle.dump(aug_search_results_test, file)

            # prep augmented lowner train + dev, clean lowner test
            tokenized_data_path = os.path.join(
                config["data_path"], f"seed_{str(seed)}",
                "02_tokenized_dataset", f"size_{gazetteer_size}",
                f"error_ratio_{error_percent_ratio}")
            os.makedirs(tokenized_data_path, exist_ok=True)

            tokenized_files = {}
            tokenized_files["aug_search_aug_train"] = get_tokenized_filepath(
                config, aug_lowner_train_path, aug_search_results_aug_train,
                tokenized_data_path)
            tokenized_files["aug_search_aug_dev"] = get_tokenized_filepath(
                config, aug_lowner_dev_path, aug_search_results_aug_dev,
                tokenized_data_path)
            tokenized_files["aug_search_test"] = get_tokenized_filepath(
                config, files["test"], aug_search_results_test,
                tokenized_data_path)

            processor = NERDataProcessor(
                config,
                get_tokenizer(config),
                tokenized_files["aug_search_aug_train"],
                tokenized_files["aug_search_aug_dev"],
                tokenized_files["aug_search_test"],
                files["types"],
                use_cache=False)
            config["num_labels"] = len(processor.labels)

            aug_search_aug_train, aug_search_aug_dev, aug_search_test = processor.get_tensor_samples(
            )
            config["train_len"] = len(aug_search_aug_train)

            # train model on augmented lowner train + validate on augmented lowner dev
            last_ckpt, best_ckpt = train_model(seed, gazetteer_size,
                                               error_percent_ratio, config,
                                               aug_search_aug_train,
                                               aug_search_aug_dev)

            # test model on augmented lowner train, dev + clean lowner test
            test_model(config, best_ckpt, last_ckpt, aug_search_aug_train,
                       "aug_search_aug_train")
            test_model(config, best_ckpt, last_ckpt, aug_search_aug_dev,
                       "aug_search_aug_dev")
            test_model(config, best_ckpt, last_ckpt, aug_search_test,
                       "aug_search_test")

            # setup database with gazetteer split
            split_search = setup_database(
                config["sent_search_algorithm"],
                config["sent_search_topk"],
                config["gaz_search_algorithm"],
                config["gaz_search_topk"],
                config["search_join_method"],
                config["search_topk"],
                reset=True,
                name=f"{seed}_{gazetteer_size}_{error_percent_ratio}_split",
                sents=sampled_multiconer,
                gazs=sampled_lownergaz)

            # get search results for clean lowner train, dev, test
            split_search_results_train = get_search_results(
                split_search, datasets["train"])
            split_search_results_dev = get_search_results(
                split_search, datasets["dev"])
            split_search_results_test = get_search_results(
                split_search, datasets["test"])

            # save split search results
            with open(
                    os.path.join(search_base_path,
                                 "split_search_results_train.pkl"),
                    "wb") as file:
                pickle.dump(split_search_results_train, file)
            with open(
                    os.path.join(search_base_path,
                                 "split_search_results_dev.pkl"),
                    "wb") as file:
                pickle.dump(split_search_results_dev, file)
            with open(
                    os.path.join(search_base_path,
                                 "split_search_results_test.pkl"),
                    "wb") as file:
                pickle.dump(split_search_results_test, file)

            # prep clean lowner train, dev, test
            tokenized_files["split_search_train"] = get_tokenized_filepath(
                config, files["train"], split_search_results_train,
                tokenized_data_path)
            tokenized_files["split_search_dev"] = get_tokenized_filepath(
                config, files["train"], split_search_results_dev,
                tokenized_data_path)
            tokenized_files["split_search_test"] = get_tokenized_filepath(
                config, files["test"], split_search_results_test,
                tokenized_data_path)

            processor = NERDataProcessor(config,
                                         get_tokenizer(config),
                                         tokenized_files["split_search_train"],
                                         tokenized_files["split_search_dev"],
                                         tokenized_files["split_search_test"],
                                         files["types"],
                                         use_cache=False)
            config["num_labels"] = len(processor.labels)

            split_search_train, split_search_dev, split_search_test = processor.get_tensor_samples(
            )
            config["train_len"] = len(split_search_train)

            # test model on clean train, dev, test
            test_model(config, best_ckpt, last_ckpt, split_search_train,
                       "split_search_train")
            test_model(config, best_ckpt, last_ckpt, split_search_dev,
                       "split_search_dev")
            test_model(config, best_ckpt, last_ckpt, split_search_test,
                       "split_search_test")

            # setup database with full gazetteer
            full_search = setup_database(config["sent_search_algorithm"],
                                         config["sent_search_topk"],
                                         config["gaz_search_algorithm"],
                                         config["gaz_search_topk"],
                                         config["search_join_method"],
                                         config["search_topk"])

            # get search results for clean lowner train, dev, test
            search_results_train = get_search_results(full_search,
                                                      datasets["train"])
            search_results_dev = get_search_results(full_search,
                                                    datasets["dev"])
            search_results_test = get_search_results(full_search,
                                                     datasets["test"])

            # save split search results
            with open(
                    os.path.join(search_base_path, "search_results_train.pkl"),
                    "wb") as file:
                pickle.dump(search_results_train, file)
            with open(os.path.join(search_base_path, "search_results_dev.pkl"),
                      "wb") as file:
                pickle.dump(search_results_dev, file)
            with open(
                    os.path.join(search_base_path, "search_results_test.pkl"),
                    "wb") as file:
                pickle.dump(search_results_test, file)

            # prep clean lowner train, dev, test
            tokenized_files["search_train"] = get_tokenized_filepath(
                config, files["train"], search_results_train,
                tokenized_data_path)
            tokenized_files["search_dev"] = get_tokenized_filepath(
                config, files["train"], search_results_dev,
                tokenized_data_path)
            tokenized_files["search_test"] = get_tokenized_filepath(
                config, files["test"], search_results_test,
                tokenized_data_path)

            processor = NERDataProcessor(config,
                                         get_tokenizer(config),
                                         tokenized_files["search_train"],
                                         tokenized_files["search_dev"],
                                         tokenized_files["search_test"],
                                         files["types"],
                                         use_cache=False)
            config["num_labels"] = len(processor.labels)

            search_train, search_dev, search_test = processor.get_tensor_samples(
            )
            config["train_len"] = len(split_search_train)

            # test model on clean train, dev, test
            test_model(config, best_ckpt, last_ckpt, search_train,
                       "full_search_train")
            test_model(config, best_ckpt, last_ckpt, search_dev,
                       "full_search_dev")
            test_model(config, best_ckpt, last_ckpt, search_test,
                       "full_search_test")