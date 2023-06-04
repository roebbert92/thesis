import copy
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

from typing import Dict, List
import torch
from torch.utils.data import DataLoader
import json
from lightning.fabric.utilities.seed import seed_everything
from haystack import Document
import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_json, tokenize_search_results_json
from models.asp_t5 import ASPT5Model, get_tokenizer
from pipelines.evaluation import factors
from configs.asp_t5 import T5_ASP_LOWNERGAZ_SENT
from hyperparameter_tuning.t5_asp_lownergaz_sent import setup_database as setup_database_lownergaz_sent
from hyperparameter_tuning.utils import get_search_results

files = {
    "types":
    os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    "train":
    os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
    "dev":
    os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
    "test":
    os.path.join(thesis_path, "data", "mlowner", "lowner_test.json"),
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

seeds = [1, 2, 3]
datasets = {"train": lowner_train, "dev": lowner_dev, "test": lowner_test}
gazetteer_sizes = [2000, 4000, 8000]
error_percent_ratios = [5, 10, 15]

config = T5_ASP_LOWNERGAZ_SENT
config.update({
    "data_path":
    os.path.join(thesis_path, "experiments", "02_content", "data")
})
parts = ["train", "dev"]


def measure_model_performance(seed: int, config,
                              search_results: Dict[str, Dict[int,
                                                             List[Document]]]):

    grad_accum_steps = factors(config["batch_size"])
    tokenizer = get_tokenizer(config)

    tokenized_data_path = os.path.join(config["data_path"],
                                       f"seed_{str(seed)}",
                                       "02_tokenized_dataset", config["name"])
    if not os.path.exists(tokenized_data_path):
        os.makedirs(tokenized_data_path)

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

    has_search = len(search_results) > 0

    tokenized_files = {}
    for part in parts:
        if has_search:
            tokenized_files[part] = tokenize_search_results_json(
                tokenizer,
                files[part],
                files["types"],
                search_results[part],
                tokenized_data_path,
                use_labels=use_labels,
                use_mentions=use_mentions,
                sent_use_labels=sent_use_labels,
                sent_use_mentions=sent_use_mentions,
                gaz_use_labels=gaz_use_labels,
                gaz_use_mentions=gaz_use_mentions,
                prepend_search_results=False)
        else:
            tokenized_files[part] = tokenize_json(tokenizer, files[part],
                                                  files["types"],
                                                  tokenized_data_path)

    processor = NERDataProcessor(
        config,
        tokenizer,
        tokenized_files["train"],
        tokenized_files["dev"],
        tokenized_files["test"] if "test" in tokenized_files else None,
        files["types"],
        use_cache=False)
    config["num_labels"] = len(processor.labels)

    train, val, test = processor.get_tensor_samples()
    config["train_len"] = len(train)

    if has_search:
        collator = NERCollator(config["train_search_dropout"], False)
    else:
        collator = ner_collate_fn

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
                                        config["name"])
    checkpoint_best = ModelCheckpoint(dirpath=checkpoint_base_path,
                                      filename="best",
                                      monitor="val_f1",
                                      mode="max",
                                      save_top_k=1)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join([str(seed), config["name"]]),
    )

    # Metrics
    metrics_base_path = os.path.join(config["data_path"], f"seed_{str(seed)}",
                                     "04_metrics", config["name"])

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
        train_val_loader = DataLoader(train,
                                      batch_size=int(
                                          train_config["batch_size"] * 4),
                                      collate_fn=ner_collate_fn,
                                      num_workers=3,
                                      persistent_workers=False,
                                      pin_memory=True,
                                      shuffle=False,
                                      prefetch_factor=20)
        dev_val_loader = DataLoader(val,
                                    batch_size=int(train_config["batch_size"] *
                                                   4),
                                    collate_fn=ner_collate_fn,
                                    num_workers=3,
                                    persistent_workers=False,
                                    pin_memory=True,
                                    shuffle=False,
                                    prefetch_factor=20)
        test_val_loader = None
        if test is not None:
            test_val_loader = DataLoader(val,
                                         batch_size=int(
                                             train_config["batch_size"] * 4),
                                         collate_fn=ner_collate_fn,
                                         num_workers=3,
                                         persistent_workers=False,
                                         pin_memory=True,
                                         shuffle=False,
                                         prefetch_factor=20)
        return model, trainer, train_loader, train_val_loader, dev_val_loader, test_val_loader

    model, trainer, train_loader, train_val_loader, dev_val_loader, test_val_loader = get_model_trainer(
    )
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
            model, trainer, train_loader, train_val_loader, dev_val_loader, test_val_loader = get_model_trainer(
            )

    def save_metrics(dataset, checkpoint):
        with open(
                os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"),
                "wb") as file:
            pickle.dump(model.test_metrics, file)

    # test last model
    trainer.test(model, train_val_loader)
    save_metrics("lowner_train", "last")
    trainer.test(model, dev_val_loader)
    save_metrics("lowner_dev", "last")
    if test_val_loader is not None:
        trainer.test(model, test_val_loader)
        save_metrics("lowner_test", "last")
    # test best model
    trainer.test(model,
                 train_val_loader,
                 ckpt_path=checkpoint_best.best_model_path)
    save_metrics("lowner_train", "best")
    trainer.test(model,
                 dev_val_loader,
                 ckpt_path=checkpoint_best.best_model_path)
    save_metrics("lowner_dev", "best")
    if test_val_loader is not None:
        trainer.test(model,
                     test_val_loader,
                     ckpt_path=checkpoint_best.best_model_path)
        save_metrics("lowner_test", "best")


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

            # create gazetteer split of multiconer + get top-5 similar gazetteers of lownergaz --> save in 00_datasets

            # create augmented lowner train, dev, + gazetteer split

            # setup database with augmented gazetteer split

            # get search results for augmented lowner train + dev, clean lowner test

            # prep augmented lowner train + dev, clean lowner test

            # train model on augmented lowner train + validate on augmented lowner dev

            # test model on augmented lowner train, dev + clean lowner test

            # setup database with gazetteer split

            # get search results for clean lowner train, dev, test

            # prep clean lowner train, dev, test

            # test model on clean train, dev, test

            # setup database with full gazetteer

            # get search results for clean lowner train, dev, test

            # prep clean lowner train, dev, test

            # test model on clean train, dev, test

        # setup database
        search = setup_database_lownergaz_sent(
            config["sent_search_algorithm"],
            config["sent_search_topk"],
            config["gaz_search_algorithm"],
            config["gaz_search_topk"],
            config["search_join_method"],
            config["search_topk"],
        )

        search_results = {}
        for part in parts:
            dataset = datasets[part]
            dataset_name = "lowner_" + part
            # save search results for augmentation
            file_name = os.path.join(config["data_path"], f"seed_{str(seed)}",
                                     "01_search_results", config['name'],
                                     f"{dataset_name}.pkl")
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))

            # get search results
            search_result = None
            if search is not None:
                if not os.path.exists(file_name):
                    search_result = get_search_results(search, dataset)
                    with open(file_name, "wb") as file:
                        pickle.dump(search_result, file)
                else:
                    with open(file_name, "rb") as file:
                        search_result = pickle.load(file)

                search_results[part] = search_result

        # train model and get results
        measure_model_performance(seed, config, search_results)
