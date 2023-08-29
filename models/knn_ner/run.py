import copy
import json
import pickle
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import argparse

from models.knn_ner.ner_trainer import NERTask
from models.knn_ner.build_datastore import build_datastore
from hyperparameter_tuning.utils import factors

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
def experiment01():
    seeds = [42]
    # use same config as flair
    for seed in seeds:
        if "PL_GLOBAL_SEED" in os.environ:
            del os.environ["PL_GLOBAL_SEED"]
        pl.seed_everything(seed)
        config = {
            "lr":
            5e-6,
            "max_epochs":
            20,
            "max_length":
            512,
            "adam_epsilon":
            1e-8,
            "weight_decay":
            0.,
            "hidden_dropout_prob":
            0.1,
            "warmup_proportion":
            0.1,
            "train_batch_size":
            40,
            "eval_batch_size":
            120,
            "accumulate_grad_batches":
            1,
            "precision":
            "bf16-mixed",
            "bert_path":
            "xlm-roberta-large",
            "file_name":
            "bio",
            "data_prefix":
            "lowner_",
            "data_dir":
            os.path.join(thesis_path, "data", "mlowner"),
            "save_ner_prediction":
            True,
            "classifier":
            "multi",
            "en_roberta":
            True,
            "gpus":
            1,
            "seed":
            seed,
            "language":
            "en",
            "lower_case":
            False,
            "data_path":
            os.path.join(thesis_path, "experiments", "01_performance", "data"),
            "fused":
            True,
            "name":
            "custom_flair_locked"
        }
        grad_accum_steps = factors(config["train_batch_size"])
        with open(os.path.join(thesis_path, "data", "mlowner",
                               "lowner_types.json"),
                  "r",
                  encoding="utf-8") as file:
            config["types"] = list(json.load(file)["entities"].keys())

        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
        torch.backends.cudnn.allow_tf32 = True  # type: ignore

        train_config = copy.deepcopy(config)

        checkpoint_base_path = os.path.join(config["data_path"],
                                            f"seed_{str(seed)}",
                                            "03_checkpoints", config["name"])
        checkpoint_best = ModelCheckpoint(dirpath=checkpoint_base_path,
                                          filename="best",
                                          save_top_k=1,
                                          monitor="val_f1",
                                          mode="max")

        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(thesis_path, "experiments", "01_performance",
                                  "lightning_logs"),
            name="_".join([str(seed), config["name"]]),
        )

        trained = False

        while not trained:
            try:

                # model = NERTask(argparse.Namespace(**train_config))
                model = NERTask.load_from_checkpoint(
                    os.path.join(checkpoint_base_path, "last.ckpt"))

                trainer = pl.Trainer(
                    accelerator="gpu",
                    logger=tb_logger,
                    devices=1,
                    log_every_n_steps=train_config["train_batch_size"] *
                    train_config["accumulate_grad_batches"],
                    accumulate_grad_batches=train_config[
                        "accumulate_grad_batches"],
                    precision=train_config["precision"],
                    max_epochs=train_config["max_epochs"],
                    check_val_every_n_epoch=1,
                    num_sanity_val_steps=0,
                    enable_checkpointing=True,
                    enable_progress_bar=True,
                    callbacks=[checkpoint_best])
                # trainer.fit(model)
                # trainer.save_checkpoint(
                #     os.path.join(checkpoint_base_path, "last.ckpt"))

                metrics_base_path = os.path.join(train_config["data_path"],
                                                 f"seed_{str(seed)}",
                                                 "04_metrics",
                                                 train_config["name"])
                os.makedirs(metrics_base_path, exist_ok=True)

                def save_metrics(dataset, checkpoint):
                    with open(
                            os.path.join(metrics_base_path,
                                         f"{checkpoint}_{dataset}.pkl"),
                            "wb") as file:
                        pickle.dump(model.test_metrics.metrics, file)

                # test last model
                trainer.test(model, model.val_train_dataloader())
                save_metrics("lowner_train", "last")
                trainer.test(model, model.val_dataloader())
                save_metrics("lowner_dev", "last")
                trainer.test(model, model.test_dataloader())
                save_metrics("lowner_test", "last")

                # test best model
                trainer.test(model,
                             model.val_train_dataloader(),
                             #ckpt_path=checkpoint_best.best_model_path)
                             ckpt_path=os.path.join(checkpoint_base_path, "best.ckpt"))
                save_metrics("lowner_train", "best")
                trainer.test(model,
                             model.val_dataloader(),
                             #ckpt_path=checkpoint_best.best_model_path)
                             ckpt_path=os.path.join(checkpoint_base_path, "best.ckpt"))
                save_metrics("lowner_dev", "best")
                trainer.test(model,
                             model.test_dataloader(),
                             #ckpt_path=checkpoint_best.best_model_path)
                             ckpt_path=os.path.join(checkpoint_base_path, "best.ckpt"))
                save_metrics("lowner_test", "best")

                trained = True
            except RuntimeError as e:
                print(e)
                train_config["accumulate_grad_batches"] = grad_accum_steps[
                    grad_accum_steps.index(
                        train_config["accumulate_grad_batches"]) + 1]
                train_config["train_batch_size"] = train_config[
                    "train_batch_size"] // train_config[
                        "accumulate_grad_batches"]


if __name__ == "__main__":
    experiment01()