import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import lightning.pytorch as pl

from models.asp_t5 import ASPT5Model, get_tokenizer
from configs.asp_t5 import T5_BASE
from data_preprocessing.tensorize import NERDataProcessor, ner_collate_fn
from torch.utils.data import DataLoader
import torch

config = T5_BASE

tokenizer = get_tokenizer(config)

processor = NERDataProcessor(config, tokenizer,
                             "datasets/wnut/wnut17train.t5-small.jsonlines",
                             "datasets/wnut/emerging.dev.t5-small.jsonlines",
                             "datasets/wnut/emerging.test.t5-small.jsonlines",
                             "datasets/wnut/wnut_types.json")
train, _, _ = processor.get_tensor_samples()
config["train_len"] = len(train)

train_loader = DataLoader(train,
                          batch_size=config["batch_size"],
                          collate_fn=ner_collate_fn,
                          num_workers=5,
                          persistent_workers=True,
                          pin_memory=True,
                          shuffle=True,
                          prefetch_factor=20)

model = ASPT5Model(config, tokenizer)

if torch.cuda.is_available():
    config["fused"] = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=True,
        devices=1,
        accumulate_grad_batches=config["gradient_accumulation_steps"],
        precision="bf16-mixed",
        max_epochs=config["num_epochs"],
        default_root_dir=thesis_path + "/experiments")

else:
    trainer = pl.Trainer(
        accelerator="cpu",
        logger=True,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        accumulate_grad_batches=config["gradient_accumulation_steps"],
        max_epochs=config["num_epochs"],
        default_root_dir=thesis_path + "/experiments")

trainer.fit(model, train_loader)
