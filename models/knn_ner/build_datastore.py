#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : build_datastore.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/10 15:54
@version: 1.0
@desc  :
"""

import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import json
import argparse
from functools import partial

from models.knn_ner.utils import collate_to_max_length
from models.knn_ner.dataset import NERDataset
from models.knn_ner.ner_trainer import NERTask

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, SequentialSampler
# from transformers import BertConfig, RobertaConfig
from transformers.models.xlm_roberta import XLMRobertaConfig

import lightning.pytorch as pl
from torch.utils.data import Subset


class Datastore(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args

        self.en_roberta = True
        self.model = NERTask.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path, batch_size=args.batch_size)
        self.entity_labels = self.model.entity_labels
        self.num_labels = len(self.entity_labels)
        self.all_outputs = []

    def forward(self, input_ids, word_maps):
        return self.model.forward(input_ids, word_maps=word_maps)

    def test_dataloader(self, ) -> DataLoader:
        dataset = NERDataset(directory=self.args.data_dir,
                             entity_labels=self.entity_labels,
                             prefix=self.args.data_prefix,
                             vocab_file=self.args.bert_path,
                             max_length=self.args.max_length,
                             file_name=self.args.file_name,
                             lower_case=self.args.lower_case,
                             language=self.args.language,
                             en_roberta=self.en_roberta)

        batch_size = self.args.batch_size
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset,
                                sampler=data_sampler,
                                batch_size=batch_size,
                                num_workers=3,
                                collate_fn=partial(collate_to_max_length,
                                                   fill_values=[0, 0, 0]),
                                drop_last=False)

        return dataloader

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.all_outputs.clear()

    def test_step(self, batch, batch_idx):
        _, word_maps, input_ids, gold_labels = batch

        features = self.forward(
            input_ids=input_ids,
            word_maps=word_maps).hidden_states  # [bsz, sent_len, feature_size]
        # [bsz, sent_len, feature_size], [bsz, sent_len], [bsz, sent_len]
        sequence_mask = torch.cat([
            torch.ones_like(word_maps[:, :1]), (word_maps[:, 1:] != 0).long()
        ],
                                  dim=1)
        word_maps[sequence_mask == 1] += 1
        word_maps = torch.nn.functional.pad(word_maps, (1, 1), value=0)
        token_lengths = torch.max(word_maps, dim=1).values
        sequence_mask = torch.zeros(word_maps.shape[0],
                                    token_lengths.max() - 1,
                                    dtype=torch.long).to(self.device)
        for i in torch.arange(int(sequence_mask.shape[0])):
            sequence_mask[i, :token_lengths[i] - 1] = 1

        self.all_outputs.append({
            "features": features.to(torch.float16).cpu(),
            "labels": gold_labels.cpu(),
            "mask": sequence_mask.bool().cpu()
        })

    def on_test_epoch_end(self):
        outputs = self.all_outputs
        hidden_size = outputs[0]['features'].shape[2]

        token_sum = sum(
            int(x['mask'].sum(dim=-1).sum(dim=-1).cpu()) for x in outputs)

        data_store_key_in_memory = np.zeros((token_sum, hidden_size),
                                            dtype=np.float16)
        data_store_val_in_memory = np.zeros((token_sum, ), dtype=np.int16)

        now_cnt = 0
        for x in outputs:
            features = x['features'].to(self.device).reshape(-1, hidden_size)
            mask = x['mask'].to(self.device)
            labels = torch.masked_select(x['labels'].to(self.device),
                                         mask).cpu().numpy().astype(np.int16)
            mask = mask.reshape(features.shape[0],
                                1).expand(features.shape[0], features.shape[1])
            features = torch.masked_select(features,
                                           mask).view(-1, hidden_size).cpu()
            np_features = features.numpy().astype(np.float16)
            data_store_key_in_memory[now_cnt:now_cnt +
                                     np_features.shape[0]] = np_features
            data_store_val_in_memory[now_cnt:now_cnt +
                                     np_features.shape[0]] = labels
            now_cnt += np_features.shape[0]

        datastore_info = {"token_sum": token_sum, "hidden_size": hidden_size}
        json.dump(datastore_info,
                  open(
                      os.path.join(self.args.datastore_path,
                                   "datastore_info.json"), "w"),
                  sort_keys=True,
                  indent=4,
                  ensure_ascii=False)

        key_file = os.path.join(self.args.datastore_path, "keys.npy")
        keys = np.memmap(key_file,
                         dtype=np.float16,
                         mode="w+",
                         shape=(token_sum, hidden_size))

        val_file = os.path.join(self.args.datastore_path, "vals.npy")
        vals = np.memmap(val_file,
                         dtype=np.int16,
                         mode="w+",
                         shape=(token_sum, ))

        keys[:] = data_store_key_in_memory[:]
        vals[:] = data_store_val_in_memory[:]

        return {"saved dir": self.args.datastore_path}


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--workers",
                        type=int,
                        default=0,
                        help="num workers for dataloader")
    parser.add_argument("--use_memory",
                        action="store_true",
                        help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--file_name",
                        default="",
                        type=str,
                        help="use for truncated sets.")
    parser.add_argument("--path_to_model_hparams_file",
                        default="",
                        type=str,
                        help="use for evaluation")
    parser.add_argument("--checkpoint_path",
                        default="",
                        type=str,
                        help="use for evaluation.")
    parser.add_argument("--datastore_path",
                        default="",
                        type=str,
                        help="use for saving datastore.")
    parser.add_argument("--en_roberta",
                        action="store_true",
                        help="whether load roberta for classification or not.")

    return parser


def build_datastore(seed: int, ckpt_name: str, ckpt_path: str, datastore_path):

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

    config = {
        "max_length": 512,
        "batch_size": 120,
        "use_memory": True,
        "bert_path": "xlm-roberta-large",
        "file_name": "bmes",
        "lower_case": False,
        "data_prefix": "lowner_train",
        "data_dir": os.path.join(thesis_path, "data", "mlowner"),
        "gpus": 1,
        "seed": seed,
        "language": "en",
        "name": "knn_ner_sent",
        "checkpoint_path": ckpt_path
    }
    knn_datastore_path = os.path.join(datastore_path,
                                      "_".join([config["name"], ckpt_name]))
    os.makedirs(knn_datastore_path, exist_ok=True)
    config["datastore_path"] = knn_datastore_path

    model = Datastore(argparse.Namespace(**config))
    trainer = pl.Trainer(accelerator="gpu",
                         logger=False,
                         devices=1,
                         precision="bf16-mixed")

    trainer.test(model)

    return knn_datastore_path

if __name__ == "__main__":
    build_datastore(
        1, "last",
        "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/03_checkpoints/knn_ner/last.ckpt",
        "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/02_tokenized_datasets/knn_ner_sent"
    )
