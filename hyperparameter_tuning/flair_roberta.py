import json
import sys
import os


thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1]
)
sys.path.append(thesis_path)

import copy
import pickle
import torch
import argparse
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from hyperparameter_tuning.utils import factors
from hyperparameter_tuning.ray_logging import TuneReportCallback
from models.flair_roberta.model import FlairModel
from ray import tune

from configs.flair import (
    FLAIR,
    FLAIR_GAZ,
    FLAIR_SENT,
    FLAIR_LOWNERGAZ,
    FLAIR_LOWNERGAZ_GAZ,
    FLAIR_GAZ_SENT,
    FLAIR_LOWNERGAZ_SENT,
)


def flair_configs():
    config = FLAIR.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["seed"] = 42
    return config, best_configs


def flair_gaz_configs():
    config = FLAIR_GAZ.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
            "train_search_dropout": 0.028260869565217374,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["seed"] = 42
    return config, best_configs


def flair_sent_configs():
    config = FLAIR_SENT.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
            "train_search_dropout": 0.21126587935893093,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["seed"] = 42
    return config, best_configs


def flair_lownergaz_configs():
    config = FLAIR_LOWNERGAZ.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
            "train_search_dropout": 0.028260869565217374,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["seed"] = 42
    return config, best_configs


def flair_gaz_sent_configs():
    config = FLAIR_GAZ_SENT.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
            "train_search_dropout": 0.05492957746478871,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["seed"] = 42
    return config, best_configs


def flair_lownergaz_gaz_configs():
    config = FLAIR_LOWNERGAZ_GAZ.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
            "train_search_dropout": 0.028260869565217374,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["seed"] = 42
    return config, best_configs


def flair_lownergaz_sent_configs():
    config = FLAIR_LOWNERGAZ_SENT.copy()
    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning", "tune")
    best_configs = [
        {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "locked_dropout_prob": 0.5,
            "hidden_dropout_prob": 0.0,
            "train_search_dropout": 0.05492957746478871,
        }
    ]
    config["lr"] = tune.uniform(5e-7, 5e-4)
    config["weight_decay"] = tune.uniform(0.0, 3e-2)
    config["locked_dropout_prob"] = tune.uniform(0.0, 0.75)
    config["hidden_dropout_prob"] = 0.0
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["seed"] = 42
    return config, best_configs


def run_training(config: dict, fixed_params: dict):
    config.update(fixed_params)

    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    pl.seed_everything(config["seed"])
    config["precision"] = "bf16-mixed"
    config["fused"] = True
    grad_accum_steps = factors(config["train_batch_size"])
    with open(
        os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
        "r",
        encoding="utf-8",
    ) as file:
        config["types"] = list(json.load(file)["entities"].keys())

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

    train_config = copy.deepcopy(config)
    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="", version=".")

    # Callbacks
    tune_report_f1 = TuneReportCallback({"f1": "val_f1"}, on=["validation_end"])

    trained = False

    while not trained:
        try:
            model = FlairModel(argparse.Namespace(**train_config))

            trainer = pl.Trainer(
                accelerator="gpu",
                logger=tb_logger,
                devices=1,
                log_every_n_steps=train_config["train_batch_size"]
                * train_config["accumulate_grad_batches"],
                accumulate_grad_batches=train_config["accumulate_grad_batches"],
                precision=train_config["precision"],
                max_epochs=train_config["max_epochs"],
                check_val_every_n_epoch=2,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[tune_report_f1],
            )
            trainer.fit(model)
            trainer.validate(model)
            trained = True
        except RuntimeError as e:
            print(e)
            train_config["accumulate_grad_batches"] = grad_accum_steps[
                grad_accum_steps.index(train_config["accumulate_grad_batches"]) + 1
            ]
            train_config["train_batch_size"] = (
                train_config["train_batch_size"]
                // train_config["accumulate_grad_batches"]
            )
