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
config["num_epochs"] = 20
config["precision"] = "bf16-mixed"  #"bf16-mixed"
config["name"] = "asp_t5_conll03"
config["batch_size"] = 32

tokenizer = get_tokenizer(config)

# processor = NERDataProcessor(config, tokenizer,
#                              "datasets/wnut/wnut17train.t5-small.jsonlines",
#                              "datasets/wnut/emerging.dev.t5-small.jsonlines",
#                              "datasets/wnut/emerging.test.t5-small.jsonlines",
#                              "datasets/wnut/wnut_types.json")

processor = NERDataProcessor(
    config, tokenizer, "datasets/conll03/conll03_train.t5-small.jsonlines",
    "datasets/conll03/conll03_dev.t5-small.jsonlines",
    "datasets/conll03/conll03_test.t5-small.jsonlines",
    "datasets/conll03/conll03_types.json")
config["num_labels"] = len(processor.labels)

train, val, test = processor.get_tensor_samples()
config["train_len"] = len(train)

train_loader = DataLoader(train,
                          batch_size=config["batch_size"],
                          collate_fn=ner_collate_fn,
                          num_workers=5,
                          persistent_workers=True,
                          pin_memory=True,
                          shuffle=True,
                          prefetch_factor=20)

val_loader = DataLoader(val,
                        batch_size=int(config["batch_size"] * 3),
                        collate_fn=ner_collate_fn,
                        num_workers=5,
                        persistent_workers=True,
                        pin_memory=True,
                        shuffle=False,
                        prefetch_factor=20)

test_loader = DataLoader(test,
                         batch_size=int(config["batch_size"] * 3),
                         collate_fn=ner_collate_fn,
                         num_workers=5,
                         persistent_workers=True,
                         pin_memory=True,
                         shuffle=False,
                         prefetch_factor=20)

if torch.cuda.is_available():
    config["fused"] = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=True,
        devices=1,
        log_every_n_steps=config["batch_size"],
        accumulate_grad_batches=config["gradient_accumulation_steps"],
        precision=config["precision"],
        max_epochs=config["num_epochs"],
        default_root_dir=thesis_path + "/experiments/" + config["name"],
        check_val_every_n_epoch=4,
        num_sanity_val_steps=0)

else:
    trainer = pl.Trainer(
        accelerator="cpu",
        logger=True,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        accumulate_grad_batches=config["gradient_accumulation_steps"],
        max_epochs=config["num_epochs"],
        default_root_dir=thesis_path + "/experiments/" + config["name"])

model = ASPT5Model(config, tokenizer)
#model = ASPT5Model.load_from_checkpoint(
#    thesis_path +
#    "/experiments/lightning_logs/version_0/checkpoints/epoch=79-step=1760.ckpt"
#)

trainer.fit(model, train_loader, val_dataloaders=val_loader)

trainer.test(model, dataloaders=test_loader)