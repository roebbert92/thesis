import copy
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from configs.asp_t5 import FLAN_T5_BASE, T5_BASE
from models.asp_t5 import ASPT5Model, get_tokenizer

from ray import tune
import torch
from torch.utils.data import DataLoader
from data_preprocessing.tensorize import NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_json
from hyperparameter_tuning.training import factors
from hyperparameter_tuning.ray_logging import TuneReportCallback
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl


def t5_asp_configs():
    config = T5_BASE.copy()
    config["asp_hidden_dim"] = 150
    config["asp_dropout_rate"] = 0.3
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["use_labels"] = True
    config["use_mentions"] = True
    config["prepend_search_results"] = False
    config["filter_exact_match"] = False
    config["filter_same_document"] = False
    config["search_data_type"] = "gazetteers"
    config["search_algorithm"] = "bm25"
    config["search_topk"] = 10
    config["seed"] = 42
    config["num_epochs"] = 24
    config["train_search_dropout"] = 0.0
    config["train_search_shuffle"] = False

    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning",
                                       "tune")
    config["name"] = "t5_asp"

    best_configs = [
        {
            "adam_weight_decay": 0.011738749999999989,
            "asp_dropout_rate": 0.4540625,
            "asp_hidden_dim": 633,
            "num_epochs": 16,
            "plm_learning_rate": 0.00017496219281663535,
            "task_learning_rate": 0.0035849253731343286,
            "warmup_ratio": 0.37917808219178084
        },
        {
            "adam_weight_decay": 0.029695833333333334,
            "asp_dropout_rate": 0.43875,
            "asp_hidden_dim": 644,
            "num_epochs": 15,
            "plm_learning_rate": 0.00020887755102040807,
            "task_learning_rate": 0.0018245454545454548,
            "warmup_ratio": 0.16076923076923078
        },
        {
            "adam_weight_decay": 0.1554625,
            "asp_dropout_rate": 0.38515625,
            "asp_hidden_dim": 428,
            "num_epochs": 11,
            "plm_learning_rate": 0.00010693877551020426,
            "task_learning_rate": 0.0027730578512396692,
            "warmup_ratio": 0.06798816568047338
        },
    ]

    config["asp_hidden_dim"] = tune.randint(100, 800)
    config["asp_dropout_rate"] = tune.uniform(0.01, 0.5)
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["use_labels"] = None
    config["use_mentions"] = None
    config["prepend_search_results"] = None
    config["filter_exact_match"] = None
    config["filter_same_document"] = None
    config["search_data_type"] = None
    config["search_algorithm"] = None
    config["search_topk"] = None
    config["seed"] = 42
    config["train_search_dropout"] = None
    config["train_search_shuffle"] = None
    config["plm_learning_rate"] = tune.uniform(5e-6, 5e-3)
    config["task_learning_rate"] = tune.uniform(1e-5, 5e-3)
    config["adam_weight_decay"] = tune.uniform(5e-4, 0.27)
    config["warmup_ratio"] = tune.uniform(0.01, 0.5)
    config["num_epochs"] = tune.randint(10, 25)

    return config, best_configs


def flan_t5_asp_configs():
    config = FLAN_T5_BASE.copy()
    config["seed"] = 42

    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning",
                                       "tune")
    config["name"] = "flan-t5_asp"

    best_configs = [
        {
            "adam_weight_decay": 0.011738749999999989,
            "asp_dropout_rate": 0.4540625,
            "asp_hidden_dim": 633,
            "num_epochs": 16,
            "plm_learning_rate": 0.00017496219281663535,
            "task_learning_rate": 0.0035849253731343286,
            "train_search_dropout": 0.05492957746478871,
            "warmup_ratio": 0.37917808219178084
        },
        {
            "adam_weight_decay": 0.12402083333333332,
            "asp_dropout_rate": 0.11718749999999999,
            "asp_hidden_dim": 342,
            "num_epochs": 21,
            "plm_learning_rate": 0.00010693877551020426,
            "task_learning_rate": 0.00413396694214876,
            "warmup_ratio": 0.29414201183431954,
        },
    ]

    config["asp_hidden_dim"] = tune.randint(100, 800)
    config["asp_dropout_rate"] = tune.uniform(0.01, 0.5)
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["use_labels"] = None
    config["use_mentions"] = None
    config["prepend_search_results"] = None
    config["filter_exact_match"] = None
    config["filter_same_document"] = None
    config["search_data_type"] = None
    config["search_algorithm"] = None
    config["search_topk"] = None
    config["seed"] = 42
    config["train_search_dropout"] = None
    config["train_search_shuffle"] = None
    config["plm_learning_rate"] = tune.uniform(5e-6, 5e-3)
    config["task_learning_rate"] = tune.uniform(1e-5, 5e-3)
    config["adam_weight_decay"] = tune.uniform(5e-4, 0.27)
    config["warmup_ratio"] = tune.uniform(0.01, 0.5)
    config["num_epochs"] = tune.randint(10, 25)

    return config, best_configs


def prep_data(path, tokenizer, config: dict):
    data_path = os.path.join(path, config["name"])
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    elif os.path.exists(
            os.path.join(data_path, "lowner_train.t5-small.jsonlines")):
        return os.path.join(data_path,
                            "lowner_train.t5-small.jsonlines"), os.path.join(
                                data_path,
                                "lowner_dev.t5-small.jsonlines"), os.path.join(
                                    thesis_path, "data", "mlowner",
                                    "lowner_types.json")
    train_file = tokenize_json(
        tokenizer,
        os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
        os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
        data_path)
    dev_file = tokenize_json(
        tokenizer,
        os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
        os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
        data_path)
    return train_file, dev_file, os.path.join(thesis_path, "data", "mlowner",
                                              "lowner_types.json")


def run_t5_asp_training(config: dict, fixed_params: dict):
    config.update(fixed_params)

    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    seed_everything(config["seed"])

    grad_accum_steps = factors(config["batch_size"])

    tokenizer = get_tokenizer(config)

    [tokenized_train_data_file, tokenized_dev_data_file,
     type_data_file] = prep_data(config["data_path"], tokenizer, config)

    processor = NERDataProcessor(config,
                                 tokenizer,
                                 tokenized_train_data_file,
                                 tokenized_dev_data_file,
                                 None,
                                 type_data_file,
                                 use_cache=False)
    config["num_labels"] = len(processor.labels)

    train, val, _ = processor.get_tensor_samples()
    config["train_len"] = len(train)

    # Callbacks
    tune_report_f1 = TuneReportCallback({"val_f1": "val_f1"},
                                        on=["validation_end"])

    config["fused"] = True
    config["precision"] = "bf16-mixed"
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="", version=".")

    train_config = copy.deepcopy(config)
    trained = False
    while not trained:
        try:
            # Train loader
            train_loader = DataLoader(train,
                                      batch_size=train_config["batch_size"],
                                      collate_fn=ner_collate_fn,
                                      num_workers=3,
                                      persistent_workers=False,
                                      pin_memory=True,
                                      shuffle=True,
                                      prefetch_factor=20)
            # Validation loaders
            val_loader = DataLoader(val,
                                    batch_size=int(train_config["batch_size"] *
                                                   4),
                                    collate_fn=ner_collate_fn,
                                    num_workers=3,
                                    persistent_workers=False,
                                    pin_memory=True,
                                    shuffle=False,
                                    prefetch_factor=20)

            trainer = pl.Trainer(
                accelerator="gpu",
                logger=tb_logger,
                devices=1,
                log_every_n_steps=train_config["batch_size"] *
                train_config["gradient_accumulation_steps"],
                accumulate_grad_batches=train_config[
                    "gradient_accumulation_steps"],
                precision=train_config["precision"],
                max_epochs=train_config["num_epochs"],
                check_val_every_n_epoch=2,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[tune_report_f1]  # type: ignore
            )

            model = ASPT5Model(train_config, tokenizer)

            trainer.fit(model, train_loader, val_dataloaders=val_loader)
            trainer.validate(model, val_loader)
            trained = True
        except Exception:
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]
