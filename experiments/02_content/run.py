import copy
import sys
import os
from preprocess_data import generate_experiment_data

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
import json
from lightning.fabric.utilities.seed import seed_everything
import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from models.asp_t5 import ASPT5Model, get_tokenizer
from pipelines.evaluation import factors
from configs.asp_t5 import T5_ASP_LOWNERGAZ_SENT
import shutil
from glob import glob


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
                erroneous_data: str, config, train: Dataset, val: Dataset):

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
                                        f"error_ratio_{error_percent_ratio}",
                                        f"error_data_{erroneous_data}")
    os.makedirs(checkpoint_base_path, exist_ok=True)
    ckpt_files = list(glob(os.path.join(checkpoint_base_path, "*.ckpt")))
    if os.path.exists(os.path.join(
            checkpoint_base_path, "last.ckpt")) and os.path.exists(
                os.path.join(checkpoint_base_path,
                             "best.ckpt")) and len(ckpt_files) == 2:
        return os.path.join(checkpoint_base_path,
                            "last.ckpt"), os.path.join(checkpoint_base_path,
                                                       "best.ckpt")

    checkpoint_best = ModelCheckpoint(dirpath=checkpoint_base_path,
                                      filename="best",
                                      monitor="val_f1",
                                      mode="max",
                                      save_top_k=1)

    for ckpt_path in ckpt_files:
        os.remove(ckpt_path)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "lightning_logs"),
        name="_".join([
            str(seed), f"size_{gazetteer_size}",
            f"error_ratio_{error_percent_ratio}",
            f"error_data_{erroneous_data}"
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
        dev_val_loader = get_validation_dataloader(config, val)
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
                                     f"error_ratio_{error_percent_ratio}",
                                     f"error_data_{erroneous_data}")
    os.makedirs(metrics_base_path, exist_ok=True)
    if os.path.exists(os.path.join(
            metrics_base_path, f"last_{name}.pkl")) and os.path.exists(
                os.path.join(metrics_base_path, f"best_{name}.pkl")):
        return
    tb_logger = TensorBoardLogger(save_dir=os.path.join(
        os.getcwd(), "lightning_logs"),
                                  name="_".join([
                                      str(seed), f"size_{gazetteer_size}",
                                      f"error_ratio_{error_percent_ratio}",
                                      f"error_data_{erroneous_data}"
                                  ]),
                                  version=0)
    tokenizer = get_tokenizer(config)
    model = ASPT5Model(config, tokenizer)
    trainer = pl.Trainer(accelerator="gpu",
                         logger=tb_logger,
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

    # check if ckpts are the same
    last_ckpt_epoch = torch.load(last_ckpt_path)["epoch"]
    best_ckpt_epoch = torch.load(best_ckpt_path)["epoch"]

    # test model
    trainer.test(model, val_loader, ckpt_path=last_ckpt_path)
    save_metrics(name, "last")
    if last_ckpt_epoch != best_ckpt_epoch:
        trainer.test(model, val_loader, ckpt_path=best_ckpt_path)
        save_metrics(name, "best")
    else:
        shutil.copy(os.path.join(metrics_base_path, f"last_{name}.pkl"),
                    os.path.join(metrics_base_path, f"best_{name}.pkl"))


if __name__ == "__main__":
    # Datapoints
    ## 00_datasets
    ## 01_search_results
    ## 02_tokenized_datasets
    ## 03_checkpoints
    ## 04_metrics
    files = {
        "types": os.path.join(thesis_path, "data", "mlowner",
                              "lowner_types.json"),
    }

    seeds = [1, 2, 3]
    gazetteer_sizes = [2000, 4000, 6000, 8000]
    error_percent_ratios = [0, 5, 10, 15]
    erroneous_data_parts = ["train", "gazetteer", "both"]

    config = T5_ASP_LOWNERGAZ_SENT
    config.update({
        "data_path":
        os.path.join(thesis_path, "experiments", "02_content", "data")
    })

    data_path = os.path.join(thesis_path, "experiments", "02_content",
                             "experiment_data_paths.json")
    if os.path.exists(data_path):
        with open(data_path) as file:
            experiment_data = json.load(file)
    else:
        experiment_data = generate_experiment_data(seeds, gazetteer_sizes,
                                                   error_percent_ratios)
    with open("experiment_data_paths.json", "w") as file:
        json.dump(experiment_data, file)

    for gazetteer_size in gazetteer_sizes:
        for error_percent_ratio in error_percent_ratios:
            for erroneous_data in erroneous_data_parts:
                if erroneous_data in ["train", "gazetteer"
                                      ] and error_percent_ratio == 0:
                    continue
                for seed in seeds:
                    # seed
                    if "PL_GLOBAL_SEED" in os.environ:
                        del os.environ["PL_GLOBAL_SEED"]
                    seed_everything(seed)
                    error_ratio = error_percent_ratio / 100
                    tokenized_files = experiment_data["02_tokenized_dataset"][
                        f"{seed}_{gazetteer_size}_{error_percent_ratio}"]

                    if erroneous_data == "both":
                        processor = NERDataProcessor(
                            config,
                            get_tokenizer(config),
                            tokenized_files["error_search_error_train"],
                            tokenized_files["error_search_error_dev"],
                            tokenized_files["error_search_test"],
                            files["types"],
                            use_cache=False)
                    elif erroneous_data == "gazetteer":
                        processor = NERDataProcessor(
                            config,
                            get_tokenizer(config),
                            tokenized_files["error_search_train"],
                            tokenized_files["error_search_dev"],
                            tokenized_files["error_search_test"],
                            files["types"],
                            use_cache=False)
                    else:
                        processor = NERDataProcessor(
                            config,
                            get_tokenizer(config),
                            tokenized_files["sampled_search_error_train"],
                            tokenized_files["sampled_search_error_dev"],
                            tokenized_files["sampled_search_test"],
                            files["types"],
                            use_cache=False)

                    config["num_labels"] = len(processor.labels)

                    error_search_error_train, error_search_error_dev, error_search_test = processor.get_tensor_samples(
                    )
                    config["train_len"] = len(error_search_error_train)

                    # train model on erroneous lowner train + validate on erroneous lowner dev
                    last_ckpt, best_ckpt = train_model(
                        seed, gazetteer_size, error_percent_ratio,
                        erroneous_data, config, error_search_error_train,
                        error_search_error_dev)

                    # test model on erroneous lowner train, dev + clean lowner test
                    test_model(config, best_ckpt, last_ckpt,
                               error_search_error_train,
                               "error_search_error_train")
                    test_model(config, best_ckpt, last_ckpt,
                               error_search_error_dev,
                               "error_search_error_dev")
                    test_model(config, best_ckpt, last_ckpt, error_search_test,
                               "error_search_test")

                    # test model on clean lowner train, dev, test with clean sampled search
                    processor = NERDataProcessor(
                        config,
                        get_tokenizer(config),
                        tokenized_files["sampled_search_train"],
                        tokenized_files["sampled_search_dev"],
                        tokenized_files["sampled_search_test"],
                        files["types"],
                        use_cache=False)
                    config["num_labels"] = len(processor.labels)

                    sampled_search_train, sampled_search_dev, sampled_search_test = processor.get_tensor_samples(
                    )
                    config["train_len"] = len(sampled_search_train)

                    # test model on clean train, dev, test
                    test_model(config, best_ckpt, last_ckpt,
                               sampled_search_train, "sampled_search_train")
                    test_model(config, best_ckpt, last_ckpt,
                               sampled_search_dev, "sampled_search_dev")
                    test_model(config, best_ckpt, last_ckpt,
                               sampled_search_test, "sampled_search_test")

                    # prep clean lowner train, dev, test
                    processor = NERDataProcessor(
                        config,
                        get_tokenizer(config),
                        tokenized_files["full_search_train"],
                        tokenized_files["full_search_dev"],
                        tokenized_files["full_search_test"],
                        files["types"],
                        use_cache=False)
                    config["num_labels"] = len(processor.labels)

                    full_search_train, full_search_dev, full_search_test = processor.get_tensor_samples(
                    )
                    config["train_len"] = len(full_search_train)

                    # test model on clean train, dev, test
                    test_model(config, best_ckpt, last_ckpt, full_search_train,
                               "full_search_train")
                    test_model(config, best_ckpt, last_ckpt, full_search_dev,
                               "full_search_dev")
                    test_model(config, best_ckpt, last_ckpt, full_search_test,
                               "full_search_test")
