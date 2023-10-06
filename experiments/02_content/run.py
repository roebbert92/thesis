import copy
from functools import partial
import sys
import os


from preprocess_data import generate_experiment_data

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2]
)
sys.path.append(thesis_path)

from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
import json
from lightning.fabric.utilities.seed import seed_everything
import pickle
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from models.flair_roberta.utils import collate_to_max_length

from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from models.flair_roberta.dataset import BIONERDataset
from models.asp_t5 import ASPT5Model, get_tokenizer
from configs.asp_t5 import T5_ASP_LOWNERGAZ_SENT
from configs.flair import FLAIR_LOWNERGAZ_SENT
import argparse
from models.flair_roberta.model import FlairModel
from hyperparameter_tuning.utils import factors
import shutil
from glob import glob


def get_t5_asp_validation_dataloader(config, dataset: Dataset):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] * 4),
        collate_fn=ner_collate_fn,
        num_workers=3,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=20,
    )


def train_t5_asp_model(
    seed: int,
    gazetteer_size: int,
    error_percent_ratio: int,
    erroneous_data: str,
    config,
    train: Dataset,
    val: Dataset,
):
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
    checkpoint_base_path = os.path.join(
        config["data_path"],
        f"seed_{str(seed)}",
        "03_checkpoints",
        f"size_{gazetteer_size}",
        f"error_ratio_{error_percent_ratio}",
        f"error_data_{erroneous_data}",
        "t5_asp",
    )
    os.makedirs(checkpoint_base_path, exist_ok=True)
    ckpt_files = list(glob(os.path.join(checkpoint_base_path, "*.ckpt")))
    if (
        os.path.exists(os.path.join(checkpoint_base_path, "last.ckpt"))
        and len(ckpt_files) == 1
    ):
        return os.path.join(checkpoint_base_path, "last.ckpt")

    for ckpt_path in ckpt_files:
        os.remove(ckpt_path)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join(
            [
                str(seed),
                f"size_{gazetteer_size}",
                f"error_ratio_{error_percent_ratio}",
                f"error_data_{erroneous_data}",
                "t5_asp",
            ]
        ),
    )

    def get_dataloaders():
        # Train loader
        train_loader = DataLoader(
            train,
            batch_size=train_config["batch_size"],
            collate_fn=collator,
            num_workers=3,
            persistent_workers=False,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=20,
        )
        # Validation loaders
        dev_val_loader = get_t5_asp_validation_dataloader(config, val)
        return train_loader, dev_val_loader

    def get_model_trainer():
        model = ASPT5Model(train_config, tokenizer)
        trainer = pl.Trainer(
            accelerator="gpu",
            logger=tb_logger,
            devices=1,
            log_every_n_steps=train_config["batch_size"]
            * train_config["gradient_accumulation_steps"],
            accumulate_grad_batches=train_config["gradient_accumulation_steps"],
            precision=train_config["precision"],
            max_epochs=train_config["num_epochs"],
            check_val_every_n_epoch=5,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        return model, trainer

    train_loader, dev_val_loader = get_dataloaders()
    model, trainer = get_model_trainer()
    while not trained:
        try:
            trainer.fit(model, train_loader, val_dataloaders=dev_val_loader)
            # save last model
            trainer.save_checkpoint(os.path.join(checkpoint_base_path, "last.ckpt"))
            trained = True
        except Exception as e:
            print(e)
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(train_config["gradient_accumulation_steps"]) + 1
            ]
            train_config["batch_size"] = (
                train_config["batch_size"]
                // train_config["gradient_accumulation_steps"]
            )
            train_loader, dev_val_loader = get_dataloaders()
            model, trainer = get_model_trainer()
            ckpt_files = list(glob(os.path.join(checkpoint_base_path, "*.ckpt")))
            for ckpt_path in ckpt_files:
                os.remove(ckpt_path)
    return os.path.join(checkpoint_base_path, "last.ckpt")


def test_t5_asp_model(
    config,
    last_ckpt_path,
    dataset: Dataset,
    name,
    seed,
    gazetteer_size,
    error_percent_ratio,
    erroneous_data,
):
    metrics_base_path = os.path.join(
        config["data_path"],
        f"seed_{str(seed)}",
        "04_metrics",
        f"size_{gazetteer_size}",
        f"error_ratio_{error_percent_ratio}",
        f"error_data_{erroneous_data}",
        "t5_asp",
    )
    os.makedirs(metrics_base_path, exist_ok=True)
    if os.path.exists(os.path.join(metrics_base_path, f"last_{name}.pkl")):
        return
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join(
            [
                str(seed),
                f"size_{gazetteer_size}",
                f"error_ratio_{error_percent_ratio}",
                f"error_data_{erroneous_data}",
                "t5_asp",
            ]
        ),
        version=0,
    )

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    config["precision"] = "bf16-mixed"
    config["fused"] = True

    tokenizer = get_tokenizer(config)
    model = ASPT5Model(config, tokenizer)
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tb_logger,
        devices=1,
        precision=config["precision"],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    val_loader = get_t5_asp_validation_dataloader(config, dataset)

    def save_metrics(dataset, checkpoint):
        with open(
            os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"), "wb"
        ) as file:
            pickle.dump(model.test_metrics, file)

    # test model
    trainer.test(model, val_loader, ckpt_path=last_ckpt_path)
    save_metrics(name, "last")


def run_t5_asp_experiment(
    experiment_data: dict,
    config: dict,
    gazetteer_size: int,
    error_percent_ratio: int,
    erroneous_data: str,
    seed: int,
):
    # seed
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    seed_everything(seed)
    tokenized_files = experiment_data["02_tokenized_dataset"][
        f"{seed}_{gazetteer_size}_{error_percent_ratio}"
    ]

    if erroneous_data == "both":
        processor = NERDataProcessor(
            config,
            get_tokenizer(config),
            tokenized_files["error_search_error_train"],
            tokenized_files["error_search_error_dev"],
            tokenized_files["error_search_test"],
            files["types"],
            use_cache=False,
        )
    elif erroneous_data == "gazetteer":
        processor = NERDataProcessor(
            config,
            get_tokenizer(config),
            tokenized_files["error_search_train"],
            tokenized_files["error_search_dev"],
            tokenized_files["error_search_test"],
            files["types"],
            use_cache=False,
        )
    else:
        processor = NERDataProcessor(
            config,
            get_tokenizer(config),
            tokenized_files["sampled_search_error_train"],
            tokenized_files["sampled_search_error_dev"],
            tokenized_files["sampled_search_test"],
            files["types"],
            use_cache=False,
        )

    config["num_labels"] = len(processor.labels)

    (
        error_search_error_train,
        error_search_error_dev,
        error_search_test,
    ) = processor.get_tensor_samples()
    config["train_len"] = len(error_search_error_train)

    # train model on erroneous lowner train + validate on erroneous lowner dev
    last_ckpt = train_t5_asp_model(
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        config,
        error_search_error_train,
        error_search_error_dev,
    )

    ## Timestep 0: Small, Erroneous gazetteer
    # test model on erroneous lowner train, dev + clean lowner test
    test_t5_asp_model(
        config,
        last_ckpt,
        error_search_error_train,
        "error_search_error_train",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )
    test_t5_asp_model(
        config,
        last_ckpt,
        error_search_error_dev,
        "error_search_error_dev",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )
    test_t5_asp_model(
        config,
        last_ckpt,
        error_search_test,
        "error_search_test",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )

    ## Timestep 1: Small, Corrected gazetteer
    # test model on clean lowner train, dev, test with clean sampled search
    processor = NERDataProcessor(
        config,
        get_tokenizer(config),
        tokenized_files["sampled_search_train"],
        tokenized_files["sampled_search_dev"],
        tokenized_files["sampled_search_test"],
        files["types"],
        use_cache=False,
    )
    config["num_labels"] = len(processor.labels)

    (
        sampled_search_train,
        sampled_search_dev,
        sampled_search_test,
    ) = processor.get_tensor_samples()
    config["train_len"] = len(sampled_search_train)

    # test model on clean train, dev, test
    test_t5_asp_model(
        config,
        last_ckpt,
        sampled_search_train,
        "sampled_search_train",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )
    test_t5_asp_model(
        config,
        last_ckpt,
        sampled_search_dev,
        "sampled_search_dev",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )
    test_t5_asp_model(
        config,
        last_ckpt,
        sampled_search_test,
        "sampled_search_test",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )

    ## Timestep 2: Full gazetteer
    # prep clean lowner train, dev, test
    processor = NERDataProcessor(
        config,
        get_tokenizer(config),
        tokenized_files["full_search_train"],
        tokenized_files["full_search_dev"],
        tokenized_files["full_search_test"],
        files["types"],
        use_cache=False,
    )
    config["num_labels"] = len(processor.labels)

    (
        full_search_train,
        full_search_dev,
        full_search_test,
    ) = processor.get_tensor_samples()
    config["train_len"] = len(full_search_train)

    # test model on clean train, dev, test
    test_t5_asp_model(
        config,
        last_ckpt,
        full_search_train,
        "full_search_train",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )
    test_t5_asp_model(
        config,
        last_ckpt,
        full_search_dev,
        "full_search_dev",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )
    test_t5_asp_model(
        config,
        last_ckpt,
        full_search_test,
        "full_search_test",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
    )


def train_flair_model(
    seed: int,
    gazetteer_size: int,
    error_percent_ratio: int,
    erroneous_data: str,
    config: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    pl.seed_everything(seed)
    config["seed"] = seed
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

    # Checkpoints
    checkpoint_base_path = os.path.join(
        config["data_path"],
        f"seed_{str(seed)}",
        "03_checkpoints",
        f"size_{gazetteer_size}",
        f"error_ratio_{error_percent_ratio}",
        f"error_data_{erroneous_data}",
        "flair",
    )
    os.makedirs(checkpoint_base_path, exist_ok=True)
    ckpt_files = list(glob(os.path.join(checkpoint_base_path, "*.ckpt")))
    if (
        os.path.exists(os.path.join(checkpoint_base_path, "last.ckpt"))
        and len(ckpt_files) == 1
    ):
        return os.path.join(checkpoint_base_path, "last.ckpt")

    for ckpt_path in ckpt_files:
        os.remove(ckpt_path)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join(
            [
                str(seed),
                f"size_{gazetteer_size}",
                f"error_ratio_{error_percent_ratio}",
                f"error_data_{erroneous_data}",
                "flair",
            ]
        ),
    )

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
                check_val_every_n_epoch=5,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            trainer.save_checkpoint(os.path.join(checkpoint_base_path, "last.ckpt"))
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
            ckpt_files = list(glob(os.path.join(checkpoint_base_path, "*.ckpt")))
            for ckpt_path in ckpt_files:
                os.remove(ckpt_path)
    return os.path.join(checkpoint_base_path, "last.ckpt")


def test_flair_model(
    config,
    last_ckpt_path,
    name,
    seed,
    gazetteer_size,
    error_percent_ratio,
    erroneous_data,
    test_dataloader: DataLoader,
):
    metrics_base_path = os.path.join(
        config["data_path"],
        f"seed_{str(seed)}",
        "04_metrics",
        f"size_{gazetteer_size}",
        f"error_ratio_{error_percent_ratio}",
        f"error_data_{erroneous_data}",
        "flair",
    )
    os.makedirs(metrics_base_path, exist_ok=True)
    if os.path.exists(os.path.join(metrics_base_path, f"last_{name}.pkl")):
        return
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join(
            [
                str(seed),
                f"size_{gazetteer_size}",
                f"error_ratio_{error_percent_ratio}",
                f"error_data_{erroneous_data}",
                "flair",
            ]
        ),
        version=0,
    )

    test_config = copy.deepcopy(config)
    test_config["precision"] = "bf16-mixed"
    test_config["fused"] = True

    model = FlairModel(argparse.Namespace(**test_config))

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tb_logger,
        devices=1,
        precision=test_config["precision"],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    def save_metrics(dataset, checkpoint):
        with open(
            os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"), "wb"
        ) as file:
            pickle.dump(model.test_metrics.metrics, file)

    # test model
    trainer.test(model, test_dataloader, ckpt_path=last_ckpt_path)
    save_metrics(name, "last")


def get_flair_dataloader(
    config, entity_labels, dataset_filepath, search_results_filepath, is_training=False
) -> DataLoader:
    kwargs = {}
    for kw in [
        "sent_use_labels",
        "sent_use_mentions",
        "gaz_use_labels",
        "gaz_use_mentions",
    ]:
        if kw in config:
            kwargs[kw] = config[kw]
    dataset = BIONERDataset(
        dataset_filepath=dataset_filepath,
        entity_labels=entity_labels,
        plm_name=config["plm_name"],
        max_length=config["max_length"],
        search_results_filepath=search_results_filepath,
        **kwargs,
    )
    kwargs = {}
    if is_training:
        batch_size = config["train_batch_size"]
        data_generator = torch.Generator()
        data_generator.manual_seed(config["seed"])
        data_sampler = RandomSampler(dataset, generator=data_generator)
        if "train_search_dropout" in config:
            kwargs["train_search_dropout"] = config["train_search_dropout"]
    else:
        batch_size = config["eval_batch_size"]
        data_sampler = SequentialSampler(dataset)

    # sampler option is mutually exclusive with shuffle
    dataloader = DataLoader(
        dataset=dataset,
        sampler=data_sampler,
        batch_size=batch_size,
        num_workers=3,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0], **kwargs),
        drop_last=False,
        persistent_workers=False,
        pin_memory=True,
    )

    return dataloader


def run_flair_experiment(
    experiment_data: dict,
    config: dict,
    gazetteer_size: int,
    error_percent_ratio: int,
    erroneous_data: str,
    seed: int,
):
    # seed
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    seed_everything(seed)
    datasets = experiment_data["00_datasets"][
        f"{seed}_{gazetteer_size}_{error_percent_ratio}"
    ]
    search_results = experiment_data["01_search_results"][
        f"{seed}_{gazetteer_size}_{error_percent_ratio}"
    ]
    with open(
        os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
        "r",
        encoding="utf-8",
    ) as file:
        config["types"] = list(json.load(file)["entities"].keys())
    entity_labels = BIONERDataset.get_labels(config["types"])
    config["seed"] = seed

    dataset_paths = {}
    search_result_paths = {}

    if erroneous_data == "both":
        dataset_paths["train"] = datasets["error_lowner_train"]
        dataset_paths["dev"] = datasets["error_lowner_dev"]
        dataset_paths["test"] = os.path.join(
            thesis_path, "data", "mlowner", "lowner_test.json"
        )
        search_result_paths["train"] = search_results[
            "error_search_results_error_train"
        ]
        search_result_paths["dev"] = search_results["error_search_results_error_dev"]
        search_result_paths["test"] = search_results["error_search_results_test"]

    elif erroneous_data == "gazetteer":
        dataset_paths["train"] = os.path.join(
            thesis_path, "data", "mlowner", "lowner_train.json"
        )
        dataset_paths["dev"] = os.path.join(
            thesis_path, "data", "mlowner", "lowner_dev.json"
        )
        dataset_paths["test"] = os.path.join(
            thesis_path, "data", "mlowner", "lowner_test.json"
        )
        search_result_paths["train"] = search_results["error_search_results_train"]
        search_result_paths["dev"] = search_results["error_search_results_dev"]
        search_result_paths["test"] = search_results["error_search_results_test"]
    else:
        dataset_paths["train"] = datasets["error_lowner_train"]
        dataset_paths["dev"] = datasets["error_lowner_dev"]
        dataset_paths["test"] = os.path.join(
            thesis_path, "data", "mlowner", "lowner_test.json"
        )
        search_result_paths["train"] = search_results[
            "sampled_search_results_error_train"
        ]
        search_result_paths["dev"] = search_results["sampled_search_results_error_dev"]
        search_result_paths["test"] = search_results["sampled_search_results_test"]

    ## Train model
    last_ckpt = train_flair_model(
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        config,
        get_flair_dataloader(
            config,
            entity_labels,
            dataset_paths["train"],
            search_result_paths["train"],
            True,
        ),
        get_flair_dataloader(
            config, entity_labels, dataset_paths["dev"], search_result_paths["dev"]
        ),
    )

    ## Timestep 0: Small, Erroneous gazetteer
    test_flair_model(
        config,
        last_ckpt,
        "error_search_error_train",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            dataset_paths["train"],
            search_result_paths["train"],
        ),
    )
    test_flair_model(
        config,
        last_ckpt,
        "error_search_error_dev",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            dataset_paths["dev"],
            search_result_paths["dev"],
        ),
    )
    test_flair_model(
        config,
        last_ckpt,
        "error_search_test",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            dataset_paths["test"],
            search_result_paths["test"],
        ),
    )

    ## Timestep 1: Small, Corrected gazetteer
    test_flair_model(
        config,
        last_ckpt,
        "sampled_search_train",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
            search_results["sampled_search_results_train"],
        ),
    )
    test_flair_model(
        config,
        last_ckpt,
        "sampled_search_dev",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
            search_results["sampled_search_results_dev"],
        ),
    )
    test_flair_model(
        config,
        last_ckpt,
        "sampled_search_test",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            os.path.join(thesis_path, "data", "mlowner", "lowner_test.json"),
            search_results["sampled_search_results_test"],
        ),
    )

    ## Timestep 2: Full gazetteer
    test_flair_model(
        config,
        last_ckpt,
        "full_search_train",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
            search_results["full_search_result_train"],
        ),
    )
    test_flair_model(
        config,
        last_ckpt,
        "full_search_dev",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
            search_results["full_search_result_dev"],
        ),
    )
    test_flair_model(
        config,
        last_ckpt,
        "full_search_test",
        seed,
        gazetteer_size,
        error_percent_ratio,
        erroneous_data,
        get_flair_dataloader(
            config,
            entity_labels,
            os.path.join(thesis_path, "data", "mlowner", "lowner_test.json"),
            search_results["full_search_result_test"],
        ),
    )
    os.remove(last_ckpt)


if __name__ == "__main__":
    # Datapoints
    ## 00_datasets
    ## 01_search_results
    ## 02_tokenized_datasets
    ## 03_checkpoints
    ## 04_metrics
    files = {
        "types": os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    }

    already_computed_path = os.path.join(
        thesis_path, "experiments", "02_content", "already_computed_runs.json"
    )
    already_computed = []
    if os.path.exists(already_computed_path):
        with open(already_computed_path, "r") as file:
            already_computed = json.load(file)

    seeds = [1, 2, 3]
    gazetteer_sizes = [2000, 4000, 8000, 16000]
    error_percent_ratios = [0, 10, 20, 30]
    erroneous_data_parts = ["train", "gazetteer", "both"]
    models = ["flair", "t5_asp"]

    data_path = os.path.join(
        thesis_path, "experiments", "02_content", "experiment_data_paths.json"
    )
    if os.path.exists(data_path):
        with open(data_path) as file:
            experiment_data = json.load(file)
    else:
        experiment_data = generate_experiment_data(
            seeds, gazetteer_sizes, error_percent_ratios
        )
    with open(data_path, "w") as file:
        json.dump(experiment_data, file)

    try:
        for gazetteer_size in gazetteer_sizes:
            for error_percent_ratio in error_percent_ratios:
                for seed in seeds:
                    for model in models:
                        if model == "t5_asp":
                            run_experiment = run_t5_asp_experiment
                            config = T5_ASP_LOWNERGAZ_SENT
                        else:
                            run_experiment = run_flair_experiment
                            config = FLAIR_LOWNERGAZ_SENT
                        config.update(
                            {
                                "data_path": os.path.join(
                                    thesis_path, "experiments", "02_content", "data"
                                )
                            }
                        )
                        if [
                            gazetteer_size,
                            error_percent_ratio,
                            erroneous_data_parts[0],
                            seed,
                            model,
                        ] in already_computed:
                            continue
                        if error_percent_ratio == 0:
                            run_experiment(
                                experiment_data,
                                config,
                                gazetteer_size,
                                error_percent_ratio,
                                erroneous_data_parts[0],
                                seed,
                            )
                            # done
                            already_computed.append(
                                [
                                    gazetteer_size,
                                    error_percent_ratio,
                                    erroneous_data_parts[0],
                                    seed,
                                    model,
                                ]
                            )
                            with open(
                                already_computed_path, "w", encoding="utf-8"
                            ) as file:
                                json.dump(already_computed, file)
                            for erroneous_data in erroneous_data_parts[1:]:
                                # copy error percent 0 over
                                if [
                                    gazetteer_size,
                                    error_percent_ratio,
                                    erroneous_data,
                                    seed,
                                    model,
                                ] in already_computed:
                                    continue
                                shutil.copytree(
                                    os.path.join(
                                        config["data_path"],
                                        f"seed_{str(seed)}",
                                        "04_metrics",
                                        f"size_{gazetteer_size}",
                                        f"error_ratio_{error_percent_ratio}",
                                        f"error_data_{erroneous_data_parts[0]}",
                                        model,
                                    ),
                                    os.path.join(
                                        config["data_path"],
                                        f"seed_{str(seed)}",
                                        "04_metrics",
                                        f"size_{gazetteer_size}",
                                        f"error_ratio_{error_percent_ratio}",
                                        f"error_data_{erroneous_data}",
                                        model,
                                    ),
                                    dirs_exist_ok=True,
                                )
                                # done
                                already_computed.append(
                                    [
                                        gazetteer_size,
                                        error_percent_ratio,
                                        erroneous_data,
                                        seed,
                                        model,
                                    ]
                                )
                                with open(
                                    already_computed_path, "w", encoding="utf-8"
                                ) as file:
                                    json.dump(already_computed, file)

                        else:
                            for erroneous_data in erroneous_data_parts:
                                if [
                                    gazetteer_size,
                                    error_percent_ratio,
                                    erroneous_data,
                                    seed,
                                    model,
                                ] in already_computed:
                                    continue
                                run_experiment(
                                    experiment_data,
                                    config,
                                    gazetteer_size,
                                    error_percent_ratio,
                                    erroneous_data,
                                    seed,
                                )
                                # done
                                already_computed.append(
                                    [
                                        gazetteer_size,
                                        error_percent_ratio,
                                        erroneous_data,
                                        seed,
                                        model,
                                    ]
                                )
                                with open(
                                    already_computed_path, "w", encoding="utf-8"
                                ) as file:
                                    json.dump(already_computed, file)

    finally:
        with open(already_computed_path, "w", encoding="utf-8") as file:
            json.dump(already_computed, file)
