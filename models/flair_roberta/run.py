import copy
import json
import pickle
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2]
)
sys.path.append(thesis_path)

import argparse

from models.flair_roberta.model import FlairModel
from hyperparameter_tuning.utils import factors

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from hyperparameter_tuning.t5_asp_lownergaz_sent import (
    setup_database as setup_database_lownergaz_sent,
)
from hyperparameter_tuning.t5_asp_gaz_sent import (
    setup_database as setup_database_gaz_sent,
)
from hyperparameter_tuning.t5_asp_lownergaz import (
    setup_database as setup_database_lownergaz,
)
from hyperparameter_tuning.t5_asp_gaz import setup_database as setup_database_gaz
from hyperparameter_tuning.t5_asp_sent import setup_database as setup_database_sent
from hyperparameter_tuning.t5_asp_lownergaz_gaz import (
    setup_database as setup_database_lownergaz_gaz,
)
from hyperparameter_tuning.utils import get_search_results, factors

from configs.asp_t5 import (
    T5_ASP_GAZ,
    T5_ASP_GAZ_SENT,
    T5_ASP_LOWNERGAZ,
    T5_ASP_LOWNERGAZ_GAZ,
    T5_ASP_LOWNERGAZ_SENT,
    T5_ASP_SENT,
)

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


def save_search_results():
    for search_name in [
        "lownergaz_sent",
        "gaz_sent",
        "lownergaz",
        "gaz",
        "sent",
        "lownergaz_gaz",
    ]:
        search = None
        if search_name == "lownergaz_sent":
            search = setup_database_lownergaz_sent(
                T5_ASP_LOWNERGAZ_SENT["sent_search_algorithm"],
                T5_ASP_LOWNERGAZ_SENT["sent_search_topk"],
                T5_ASP_LOWNERGAZ_SENT["gaz_search_algorithm"],
                T5_ASP_LOWNERGAZ_SENT["gaz_search_topk"],
                T5_ASP_LOWNERGAZ_SENT["search_join_method"],
                T5_ASP_LOWNERGAZ_SENT["search_topk"],
            )
        if search_name == "gaz_sent":
            search = setup_database_gaz_sent(
                T5_ASP_GAZ_SENT["sent_search_algorithm"],
                T5_ASP_GAZ_SENT["sent_search_topk"],
                T5_ASP_GAZ_SENT["gaz_search_algorithm"],
                T5_ASP_GAZ_SENT["gaz_search_topk"],
                T5_ASP_GAZ_SENT["search_join_method"],
                T5_ASP_GAZ_SENT["search_topk"],
            )
        if search_name == "lownergaz":
            search = setup_database_lownergaz(
                T5_ASP_LOWNERGAZ["search_algorithm"], T5_ASP_LOWNERGAZ["search_topk"]
            )
        if search_name == "gaz":
            search = setup_database_gaz(
                T5_ASP_GAZ["search_algorithm"], T5_ASP_GAZ["search_topk"]
            )
        if search_name == "sent":
            search = setup_database_sent(
                T5_ASP_SENT["search_algorithm"], T5_ASP_SENT["search_topk"]
            )

        if search_name == "lownergaz_gaz":
            search = setup_database_lownergaz_gaz(
                T5_ASP_LOWNERGAZ_GAZ["search_algorithm"],
                T5_ASP_LOWNERGAZ_GAZ["search_topk"],
            )

        # go through all datasets
        if search is not None:
            for part in ["train", "dev", "test"]:
                dataset_name = "lowner_" + part
                # save search results for augmentation
                file_name = os.path.join(
                    os.path.join(thesis_path, "experiments", "01_performance", "data"),
                    "01_search_results",
                    search_name,
                    f"{dataset_name}.pkl",
                )
                if not os.path.exists(os.path.dirname(file_name)):
                    os.makedirs(os.path.dirname(file_name))

                # get search results
                search_result = None
                if search is not None:
                    if not os.path.exists(file_name):
                        with open(
                            os.path.join(
                                os.path.join(thesis_path, "data", "mlowner"),
                                dataset_name + ".json",
                            ),
                            "r",
                            encoding="utf-8",
                        ) as file:
                            dataset = json.load(file)
                        search_result = get_search_results(search, dataset)
                        with open(file_name, "wb") as file:
                            pickle.dump(search_result, file)
                    else:
                        with open(file_name, "rb") as file:
                            search_result = pickle.load(file)


def experiment01(config: dict):
    seeds = [42]
    for seed in seeds:
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

        checkpoint_base_path = os.path.join(
            config["data_path"], f"seed_{str(seed)}", "03_checkpoints", config["name"]
        )
        checkpoint_best = ModelCheckpoint(
            dirpath=checkpoint_base_path,
            filename="best",
            save_top_k=1,
            monitor="val_f1",
            mode="max",
        )

        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(
                thesis_path, "experiments", "01_performance", "lightning_logs"
            ),
            name="_".join([str(seed), config["name"]]),
        )

        trained = False

        while not trained:
            try:
                model = FlairModel(argparse.Namespace(**train_config))
                # model = FlairModel.load_from_checkpoint(
                #     os.path.join(checkpoint_base_path, "last.ckpt"))

                trainer = pl.Trainer(
                    accelerator="gpu",
                    logger=tb_logger,
                    devices=1,
                    log_every_n_steps=train_config["train_batch_size"]
                    * train_config["accumulate_grad_batches"],
                    accumulate_grad_batches=train_config["accumulate_grad_batches"],
                    precision=train_config["precision"],
                    max_epochs=train_config["max_epochs"],
                    check_val_every_n_epoch=1,
                    num_sanity_val_steps=0,
                    enable_checkpointing=True,
                    enable_progress_bar=True,
                    callbacks=[checkpoint_best],
                )
                trainer.fit(model)
                trainer.save_checkpoint(os.path.join(checkpoint_base_path, "last.ckpt"))

                metrics_base_path = os.path.join(
                    train_config["data_path"],
                    f"seed_{str(seed)}",
                    "04_metrics",
                    train_config["name"],
                )
                os.makedirs(metrics_base_path, exist_ok=True)

                def save_metrics(dataset, checkpoint):
                    with open(
                        os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"),
                        "wb",
                    ) as file:
                        pickle.dump(model.test_metrics.metrics, file)

                # test last model
                trainer.test(model, model.val_train_dataloader())
                save_metrics("lowner_train", "last")
                trainer.test(model, model.val_dataloader())
                save_metrics("lowner_dev", "last")
                trainer.test(model, model.test_dataloader())
                save_metrics("lowner_test", "last")

                # test best model
                trainer.test(
                    model,
                    model.val_train_dataloader(),
                    ckpt_path=os.path.join(checkpoint_base_path, "best.ckpt"),
                )
                save_metrics("lowner_train", "best")
                trainer.test(
                    model,
                    model.val_dataloader(),
                    ckpt_path=os.path.join(checkpoint_base_path, "best.ckpt"),
                )
                save_metrics("lowner_dev", "best")
                trainer.test(
                    model,
                    model.test_dataloader(),
                    ckpt_path=os.path.join(checkpoint_base_path, "best.ckpt"),
                )
                save_metrics("lowner_test", "best")

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


if __name__ == "__main__":
    save_search_results()
