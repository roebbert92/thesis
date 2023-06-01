from collections import defaultdict
import copy
import sys
import os

thesis_path = '/home/loebbert/projects/thesis'
sys.path.append(thesis_path)

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_json, tokenize_search_results_json
from models.asp_t5 import ASPT5Model, get_tokenizer
from pipelines.evaluation import factors
from configs.asp_t5 import T5_BASE, FLAN_T5_BASE
import json
from hyperparameter_tuning.t5_asp_lownergaz_sent import setup_database as setup_database_lownergaz_sent
from hyperparameter_tuning.t5_asp_gaz_sent import setup_database as setup_database_gaz_sent
from hyperparameter_tuning.t5_asp_lownergaz import setup_database as setup_database_lownergaz
from hyperparameter_tuning.t5_asp_gaz import setup_database as setup_database_gaz
from hyperparameter_tuning.t5_asp_sent import setup_database as setup_database_sent
from hyperparameter_tuning.utils import get_search_results
from lightning.fabric.utilities.seed import seed_everything
from haystack import Document
import pickle
from data_metrics.entity_coverage_ratio import entity_coverage_ratio
from data_metrics.sample_similarity import get_search_sample_similarity
import pandas as pd
from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

t5_asp_lownergaz_sent = T5_BASE.copy()
t5_asp_lownergaz_sent.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "gaz_search_algorithm": "bm25",
    "gaz_search_topk": 6,
    "gaz_use_labels": True,
    "gaz_use_mentions": False,
    "num_epochs": 16,
    "plm_learning_rate": 0.00017496219281663535,
    "search_join_method": "reciprocal_rank_fusion",
    "search_topk": 8,
    "sent_search_algorithm": "ann",
    "sent_search_topk": 6,
    "sent_use_labels": True,
    "sent_use_mentions": True,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_lownergaz_sent"
})

t5_asp_lownergaz = FLAN_T5_BASE.copy()
t5_asp_lownergaz.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "num_epochs": 16,
    "plm_learning_rate": 0.00017496219281663535,
    "search_algorithm": "bm25",
    "search_topk": 8,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "use_labels": True,
    "use_mentions": False,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_lownergaz"
})

t5_asp_gaz = FLAN_T5_BASE.copy()
t5_asp_gaz.update({
    "adam_weight_decay": 0.018862500000000015,
    "asp_dropout_rate": 0.43875,
    "asp_hidden_dim": 799,
    "num_epochs": 17,
    "plm_learning_rate": 0.00020887755102040807,
    "search_algorithm": "bm25",
    "search_topk": 6,
    "task_learning_rate": 0.003949473684210526,
    "train_search_dropout": 0.028260869565217374,
    "use_labels": True,
    "use_mentions": False,
    "warmup_ratio": 0.20864864864864865,
    "name": "t5_asp_gaz"
})

t5_asp_gaz_sent = T5_BASE.copy()
t5_asp_gaz_sent.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "gaz_search_algorithm": "bm25",
    "gaz_search_topk": 6,
    "gaz_use_labels": True,
    "gaz_use_mentions": False,
    "num_epochs": 24,
    "plm_learning_rate": 0.00017496219281663535,
    "search_join_method": "reciprocal_rank_fusion",
    "search_topk": 8,
    "sent_search_algorithm": "ann",
    "sent_search_topk": 6,
    "sent_use_labels": True,
    "sent_use_mentions": True,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_gaz_sent"
})

t5_asp_sent = T5_BASE.copy()
t5_asp_sent.update({
    "adam_weight_decay": 0.49637507889057786,
    "asp_dropout_rate": 0.3,
    "asp_hidden_dim": 142,
    "num_epochs": 20,
    "plm_learning_rate": 5e-05,
    "search_algorithm": "ann",
    "search_topk": 8,
    "task_learning_rate": 0.0013480523331922776,
    "train_search_dropout": 0.21126587935893093,
    "use_labels": True,
    "use_mentions": True,
    "warmup_ratio": 0.184451637360714,
    "name": "t5_asp_sent"
})

t5_asp = T5_BASE.copy()
t5_asp.update({
    "adam_weight_decay": 0.12402083333333332,
    "asp_dropout_rate": 0.11718749999999999,
    "asp_hidden_dim": 342,
    "num_epochs": 21,
    "plm_learning_rate": 0.00010693877551020426,
    "task_learning_rate": 0.00413396694214876,
    "warmup_ratio": 0.29414201183431954,
    "name": "t5_asp"
})

with open("/home/loebbert/projects/thesis/data/mlowner/lowner_train.json",
          encoding="utf-8") as file:
    lowner_train = json.load(file)

with open("/home/loebbert/projects/thesis/data/mlowner/lowner_dev.json",
          encoding="utf-8") as file:
    lowner_dev = json.load(file)

with open("/home/loebbert/projects/thesis/data/mlowner/lowner_test.json",
          encoding="utf-8") as file:
    lowner_test = json.load(file)

files = {
    "types": os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    "train": os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
    "dev": os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
    "test": os.path.join(thesis_path, "data", "mlowner", "lowner_test.json"),
}

seeds = [1, 2, 3]
datasets = {"train": lowner_train, "dev": lowner_dev, "test": lowner_test}
configs = [
    t5_asp_lownergaz_sent, t5_asp_lownergaz, t5_asp_gaz, t5_asp_gaz_sent,
    t5_asp_sent, t5_asp
]
for config in configs:
    config.update({
        "data_path":
        os.path.join(thesis_path, "experiments", "01_performance",
                     "tokenized_files")
    })
parts = ["train", "dev"]

search_result_eecr = []
search_result_ccr = []

model_metrics = []


def get_model_performance(seed: int, config,
                          search_results: Dict[str, Dict[int,
                                                         List[Document]]]):

    grad_accum_steps = factors(config["batch_size"])
    tokenizer = get_tokenizer(config)

    data_path = os.path.join(config["data_path"], str(seed), config["name"])
    if not os.path.exists(data_path):
        os.makedirs(data_path)

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
                data_path,
                use_labels=use_labels,
                use_mentions=use_mentions,
                sent_use_labels=sent_use_labels,
                sent_use_mentions=sent_use_mentions,
                gaz_use_labels=gaz_use_labels,
                gaz_use_mentions=gaz_use_mentions,
                prepend_search_results=False)
        else:
            tokenized_files[part] = tokenize_json(tokenizer, files[part],
                                                  files["types"], data_path)

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
    checkpoint_base_path = os.path.join(thesis_path, "experiments",
                                        "01_performance", "checkpoints",
                                        str(seed), config["name"])
    checkpoint_best = ModelCheckpoint(dirpath=checkpoint_base_path,
                                      filename="best",
                                      monitor="val_f1",
                                      mode="max",
                                      save_top_k=1)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join([str(seed), config["name"]]),
    )

    metrics = []

    while not trained:
        try:
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
                                        batch_size=int(
                                            train_config["batch_size"] * 4),
                                        collate_fn=ner_collate_fn,
                                        num_workers=3,
                                        persistent_workers=False,
                                        pin_memory=True,
                                        shuffle=False,
                                        prefetch_factor=20)
            test_val_loader = None
            if test is not None:
                test_val_loader = DataLoader(
                    val,
                    batch_size=int(train_config["batch_size"] * 4),
                    collate_fn=ner_collate_fn,
                    num_workers=3,
                    persistent_workers=False,
                    pin_memory=True,
                    shuffle=False,
                    prefetch_factor=20)

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

            model = ASPT5Model(train_config, tokenizer)

            trainer.fit(model, train_loader, val_dataloaders=dev_val_loader)
            # save last model
            trainer.save_checkpoint(
                os.path.join(checkpoint_base_path, "last.ckpt"))

            def save_metrics(dataset, checkpoint):
                res = model.test_metrics.metrics_per_sample()
                res["seed"] = seed
                res["dataset"] = dataset
                res["checkpoint"] = checkpoint
                metrics.extend(res.to_dict(orient="records"))

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
            trained = True
        except Exception as e:
            print(e)
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]
    metrics_df = pd.DataFrame.from_records(metrics)
    metrics_df["model"] = config["name"]
    return metrics_df


for seed in seeds:
    # seed
    seed_everything(seed)

    for config in configs:
        # setup database
        search = None
        if config["name"] == "t5_asp_lownergaz_sent":
            search = setup_database_lownergaz_sent(
                config["sent_search_algorithm"],
                config["sent_search_topk"],
                config["gaz_search_algorithm"],
                config["gaz_search_topk"],
                config["search_join_method"],
                config["search_topk"],
            )
        if config["name"] == "t5_asp_gaz_sent":
            search = setup_database_gaz_sent(
                config["sent_search_algorithm"],
                config["sent_search_topk"],
                config["gaz_search_algorithm"],
                config["gaz_search_topk"],
                config["search_join_method"],
                config["search_topk"],
            )
        if config["name"] == "t5_asp_lownergaz":
            search = setup_database_lownergaz(config["search_algorithm"],
                                              config["search_topk"])
        if config["name"] == "t5_asp_gaz":
            search = setup_database_gaz(config["search_algorithm"],
                                        config["search_topk"])
        if config["name"] == "t5_asp_sent":
            search = setup_database_sent(config["search_algorithm"],
                                         config["search_topk"])

        # go through all datasets
        search_results = {}
        for part in parts:
            dataset = datasets[part]
            dataset_name = "lowner_" + part
            # save search results for augmentation
            file_name = os.path.join(
                thesis_path, "experiments", "01_performance", "search_results",
                f"{str(seed)}_{config['name']}_{dataset_name}.pkl")

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
                # calculate the expected entity coverage ratio for each sample
                for idx, item in tqdm(enumerate(dataset),
                                      total=len(dataset),
                                      desc="EECR"):
                    results = [doc.to_dict() for doc in search_result[idx]]
                    _, _, eecr, c = entity_coverage_ratio(results, [item])
                    search_result_eecr.append({
                        "dataset": dataset_name,
                        "model": config["name"],
                        "doc_id": item["doc_id"],
                        "seed": seed,
                        "eecr": eecr,
                        "targets": c
                    })

                # calculate the context coverage ratio for each sample
                for result in tqdm(get_search_sample_similarity(
                        dataset, search_result),
                                   total=len(dataset),
                                   desc="CCR"):
                    res = {
                        "dataset": dataset_name,
                        "model": config["name"],
                        "seed": seed,
                    }
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            res[key] = float(value.cpu().numpy())
                        else:
                            res[key] = value
                    search_result_ccr.append(res)

        # train model and get results
        model_metrics.append(
            get_model_performance(seed, config, search_results))

df = pd.DataFrame.from_records(search_result_eecr)
file_name = os.path.join("/home/loebbert/projects/thesis", "experiments",
                         "01_performance", "search_result_eecr_df.pkl")
df.to_pickle(file_name)

file_name = os.path.join("/home/loebbert/projects/thesis", "experiments",
                         "01_performance", "search_result_ccr_df.pkl")
df = pd.DataFrame.from_records(search_result_ccr)
df.to_pickle(file_name)

model_metrics_df = pd.concat(model_metrics)
file_name = os.path.join("/home/loebbert/projects/thesis", "experiments",
                         "01_performance", "model_metrics_df.pkl")
model_metrics_df.to_pickle(file_name)
