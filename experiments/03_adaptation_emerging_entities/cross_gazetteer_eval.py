import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2]
)
sys.path.append(thesis_path)

import shutil
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data_preprocessing.tensorize import NERDataProcessor, ner_collate_fn
from models.asp_t5 import ASPT5Model, get_tokenizer
from configs.asp_t5 import BEST_WNUT_T5_ASP_LOWNERGAZ_SENT

eval_gazetteer_contents = {
    2: [
        ("wnut_train", "wnut_dev"),
        ("lownergaz_sent", "wnut_train", "wnut_dev"),
        ("lownergaz_sent",),
    ],
    3: [
        ("wnut_train", "wnut_dev", "wnut_test"),
        ("lownergaz_sent", "wnut_train", "wnut_dev", "wnut_test"),
        ("lownergaz_sent",),
    ],
}

train_gazetteer_contents = [
    ("lownergaz_sent",),
    ("wnut_train",),
    ("lownergaz_sent", "wnut_train"),
]

seeds = [1, 2, 3]

files = {
    "types": os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    "wnut_test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
}

gazetteer, finetuning, pretrained, config = (
    True,
    "full",
    True,
    BEST_WNUT_T5_ASP_LOWNERGAZ_SENT,
)
config.update(
    {
        "data_path": os.path.join(
            thesis_path, "experiments", "03_adaptation_emerging_entities", "data"
        )
    }
)


def get_validation_dataloader(config, dataset: Dataset):
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] * 2),
        collate_fn=ner_collate_fn,
        num_workers=3,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=20,
    )


def test_model(config, best_ckpt_path, last_ckpt_path, dataset: Dataset, name):
    metrics_base_path = os.path.join(
        config["data_path"],
        f"seed_{str(seed)}",
        "05_cross_gazetteer_eval",
        f"{train_gaz_idx}_{eval_gaz_idx}_{timestep}",
    )
    os.makedirs(metrics_base_path, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(
            thesis_path,
            "experiments",
            "03_adaptation_emerging_entities",
            "lightning_logs",
        ),
        name="_".join(
            [
                str(seed),
                f"eval_{gazetteer}_{finetuning}_{pretrained}_{config['name']}",
                str(train_gaz_idx),
            ]
        ),
        version=f"{str(eval_gaz_idx)}_{timestep}" + name,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tb_logger,
        devices=1,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    val_loader = get_validation_dataloader(config, dataset)

    def save_metrics(model, dataset, checkpoint):
        with open(
            os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"), "wb"
        ) as file:
            pickle.dump(model.test_metrics, file)

    last_ckpt_epoch = torch.load(last_ckpt_path)["epoch"] - 1
    best_ckpt_epoch = torch.load(best_ckpt_path)["epoch"]

    # test model
    last_model = ASPT5Model.load_from_checkpoint(last_ckpt_path)
    trainer.test(last_model, val_loader)
    save_metrics(last_model, name, "last")
    if last_ckpt_epoch != best_ckpt_epoch:
        best_model = ASPT5Model.load_from_checkpoint(best_ckpt_path)
        trainer.test(best_model, val_loader)
        save_metrics(best_model, name, "best")
    else:
        shutil.copy(
            os.path.join(metrics_base_path, f"last_{name}.pkl"),
            os.path.join(metrics_base_path, f"best_{name}.pkl"),
        )


for timestep, eval_gaz in eval_gazetteer_contents.items():
    wnut_only = NERDataProcessor(
        config,
        get_tokenizer(config),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            f"1_{timestep}",
            "wnut_train.t5-small.jsonlines",
        ),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            f"1_{timestep}",
            "wnut_dev.t5-small.jsonlines",
        ),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            f"1_{timestep}",
            "wnut_test.t5-small.jsonlines",
        ),
        files["types"],
        use_cache=False,
    )

    lownergaz_wnut = NERDataProcessor(
        config,
        get_tokenizer(config),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            f"2_{timestep}",
            "wnut_train.t5-small.jsonlines",
        ),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            f"2_{timestep}",
            "wnut_dev.t5-small.jsonlines",
        ),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            f"2_{timestep}",
            "wnut_test.t5-small.jsonlines",
        ),
        files["types"],
        use_cache=False,
    )

    lownergaz_only = NERDataProcessor(
        config,
        get_tokenizer(config),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            "0_0",
            "wnut_train.t5-small.jsonlines",
        ),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            "0_0",
            "wnut_dev.t5-small.jsonlines",
        ),
        os.path.join(
            config["data_path"],
            "02_tokenized_dataset",
            "ann_6_bm25_12_reciprocal_rank_fusion_16",
            "0_0",
            "wnut_test.t5-small.jsonlines",
        ),
        files["types"],
        use_cache=False,
    )
    for seed in seeds:
        for train_gaz_idx, train_gaz in enumerate(train_gazetteer_contents):
            checkpoint_base_path = os.path.join(
                config["data_path"],
                f"seed_{str(seed)}",
                "03_checkpoints",
                f"{gazetteer}_{finetuning}_{pretrained}_{config['name']}",
                f"{train_gaz_idx}",
            )
            best_ckpt_path = os.path.join(checkpoint_base_path, "best.ckpt")
            last_ckpt_path = os.path.join(checkpoint_base_path, "last.ckpt")
            for eval_gaz_idx, _ in enumerate(eval_gaz):
                if eval_gaz_idx == 0:
                    (
                        train_dataset,
                        dev_dataset,
                        test_dataset,
                    ) = wnut_only.get_tensor_samples()
                elif eval_gaz_idx == 1:
                    (
                        train_dataset,
                        dev_dataset,
                        test_dataset,
                    ) = lownergaz_wnut.get_tensor_samples()
                else:
                    (
                        train_dataset,
                        dev_dataset,
                        test_dataset,
                    ) = lownergaz_only.get_tensor_samples()
                test_model(
                    config, best_ckpt_path, last_ckpt_path, test_dataset, "wnut_test"
                )
