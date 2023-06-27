#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : knn_ner_trainer.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/10 16:03
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
from metrics import SpanF1ForNER
from ner_trainer import NERTask

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from tqdm import tqdm

import lightning.pytorch as pl


class KNNNERTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args

        self.en_roberta = True

        self.model = NERTask.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path, batch_size=args.batch_size)

        self.entity_labels = self.model.entity_labels
        self.num_labels = len(self.entity_labels)

        self.test_metrics = SpanF1ForNER(entity_labels=self.entity_labels)

        self.cosine = torch.nn.CosineSimilarity(dim=-1)

        info = json.load(
            open(os.path.join(self.args.datastore_path, "datastore_info.json"),
                 "r"))
        key_file = os.path.join(self.args.datastore_path, "keys.npy")
        keys = np.memmap(key_file,
                         dtype=np.float16,
                         mode="r",
                         shape=(info['token_sum'], info['hidden_size']))
        # keys_in_memory = np.zeros((info['token_sum'], info['hidden_size']),
        #                           dtype=np.float16)
        keys_in_memory = np.zeros((20000, info['hidden_size']),
                                  dtype=np.float16)
        keys_in_memory[:] = keys[:20000]

        self.keys = torch.from_numpy(keys_in_memory)

        val_file = os.path.join(self.args.datastore_path, "vals.npy")
        vals = np.memmap(val_file,
                         dtype=np.int16,
                         mode="r",
                         shape=(info['token_sum'], ))
        #vals_in_memory = np.zeros((info['token_sum'], ), dtype=np.int64)
        vals_in_memory = np.zeros((20000, ), dtype=np.int64)
        vals_in_memory[:] = vals[:20000]

        self.vals = torch.from_numpy(vals_in_memory)

        self.link_temperature = torch.tensor(self.args.link_temperature)

        self.link_ratio = torch.tensor(self.args.link_ratio)

        self.keys = self.keys.cuda()  # [token_num, feature_size]
        self.norm_1 = (self.keys**2).sum(dim=1,
                                         keepdim=True).sqrt()  # [token_num, 1]
        self.vals = self.vals.cuda()
        self.link_temperature = self.link_temperature.cuda()
        self.link_ratio = self.link_ratio.cuda()

    def forward(self, input_ids, word_maps, labels=None):
        return self.model.forward(input_ids, word_maps, labels=labels)

    def test_step(self, batch, batch_idx):
        idxs, word_maps, input_ids, gold_labels = batch

        bert_classifiaction_outputs = self.forward(input_ids=input_ids,
                                                   word_maps=word_maps)

        argmax_labels = self.postprocess_logits_to_labels(
            bert_classifiaction_outputs.logits.to(torch.float16),
            bert_classifiaction_outputs.hidden_states.to(torch.float16))
        self.test_metrics.update(idxs, word_maps, argmax_labels, gold_labels)

    def on_test_epoch_end(self):
        errors = self.test_metrics.metrics.errors()
        f1 = self.test_metrics.metrics.f1()
        precision = self.test_metrics.metrics.precision()
        recall = self.test_metrics.metrics.recall()
        self.log_dict(
            {
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "test_error_type1": errors[0],
                "test_error_type2": errors[1],
                "test_error_type3": errors[2],
                "test_error_type4": errors[3],
                "test_error_type5": errors[4],
            },
            logger=True,
            on_epoch=True)
        super().on_test_epoch_end()

    def postprocess_logits_to_labels(self, logits, hidden):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(
            logits, dim=2)  # shape of [batch_size, seq_len, num_labels]

        batch_size = hidden.shape[0]
        sent_len = hidden.shape[1]
        hidden_size = hidden.shape[-1]
        token_num = self.keys.shape[0]

        # cosine similarity
        batch_sim_size = 10000
        batch_query_size = 30
        hidden = hidden.view(-1, hidden_size)  # [bsz*sent_len, feature_size]
        query_batches = hidden.shape[0] // batch_query_size
        key_batches = self.keys.shape[0] // batch_sim_size
        scores = torch.empty((hidden.shape[0], self.args.topk),
                             device=self.device)
        top_idxs = torch.empty((hidden.shape[0], self.args.topk),
                               dtype=torch.int64,
                               device=self.device)
        for idx, x in tqdm(enumerate(
                torch.tensor_split(hidden, query_batches, dim=0)),
                           total=query_batches):
            start_query = idx * batch_query_size
            end_query = start_query + batch_query_size
            for batch_idx, right in enumerate(
                    torch.tensor_split(self.keys, key_batches, dim=0)):
                left = x.unsqueeze(dim=1).repeat((1, right.shape[0], 1))
                right = right.unsqueeze(dim=0).repeat((left.shape[0], 1, 1))
                sim = self.cosine(left, right)
                start_key = batch_idx * batch_sim_size
                idxs = torch.arange(
                    start_key, start_key + right.shape[1]).unsqueeze(0).repeat(
                        (left.shape[0], 1)).long().to(self.device)
                if batch_idx > 0:
                    scores[start_query:end_query], indices = torch.topk(
                        torch.cat([scores[start_query:end_query], sim], dim=1),
                        self.args.topk)
                    top_idxs[start_query:end_query] = torch.cat(
                        [top_idxs[start_query:end_query], idxs],
                        dim=1).gather(1, index=indices)
                else:
                    scores[start_query:end_query], top_idxs[
                        start_query:end_query] = torch.topk(
                            sim, self.args.topk)

        scores = scores.view(batch_size, sent_len, -1)  # [bsz, sent_len, topk]
        knn_labels = self.vals.view(1, token_num).expand(
            hidden.shape[0],
            token_num).gather(dim=-1,
                              index=top_idxs).view(batch_size, sent_len,
                                                   -1)  # [bsz, sent_len, topk]

        # if (self.args.topk != -1 and scores.shape[-1] > self.args.topk):
        #     topk_scores, topk_idxs = torch.topk(
        #         scores, dim=-1, k=self.args.topk)  # [bsz, sent_len, topk]
        #     scores = topk_scores
        #     knn_labels = knn_labels.gather(
        #         dim=-1, index=topk_idxs)  # [bsz, sent_len, topk]

        sim_probs = torch.softmax(scores / self.link_temperature,
                                  dim=-1)  # [bsz, sent_len, token_num]

        knn_probabilities = torch.zeros_like(
            sim_probs[:, :, 0]).unsqueeze(-1).repeat(
                [1, 1, self.num_labels])  # [bsz, sent_len, num_labels]
        knn_probabilities = knn_probabilities.scatter_add(
            dim=2, index=knn_labels,
            src=sim_probs)  # [bsz, sent_len, num_labels]

        probabilities = self.link_ratio * knn_probabilities + (
            1 - self.link_ratio) * probabilities

        argmax_labels = torch.argmax(probabilities, 2,
                                     keepdim=False)  # [bsz, sent_len]
        return argmax_labels


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
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--file_name",
                        default="",
                        type=str,
                        help="use for truncated sets.")
    parser.add_argument("--save_ner_prediction",
                        action="store_true",
                        help="only work for test.")
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
    parser.add_argument("--link_temperature",
                        default=1.0,
                        type=float,
                        help="temperature used by edge linking.")
    parser.add_argument("--link_ratio",
                        default=0.0,
                        type=float,
                        help="ratio of vocab probs predicted by edge linking.")
    parser.add_argument(
        "--topk",
        default=64,
        type=int,
        help=
        "use topk-scored neighbor tgt nodes for link prediction and probability compuation."
    )
    parser.add_argument("--en_roberta",
                        action="store_true",
                        help="whether load roberta for classification or not.")

    return parser


def test_knn_ner(seed: int, ckpt_name: str, ckpt_path: str,
                 datastore_path: str, dataloader: DataLoader):
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    config = {
        "batch_size": 5,
        "seed": seed,
        "checkpoint_path": ckpt_path,
        "datastore_path": datastore_path,
        "link_temperature": 0.013,
        "link_ratio": 0.32,
        "topk": 256,
        "name": "_".join(["knn_ner_sent", ckpt_name])
    }
    model = KNNNERTask(argparse.Namespace(**config))

    trainer = pl.Trainer(accelerator="gpu",
                         logger=False,
                         devices=1,
                         precision="bf16-mixed")

    trainer.test(model, dataloader)

    return model.test_metrics.metrics


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    model = NERTask.load_from_checkpoint(
        "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/03_checkpoints/knn_ner/last.ckpt"
    )
    dev_dataloader = model.val_dataloader()

    test_knn_ner(
        1, "last",
        "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/03_checkpoints/knn_ner/last.ckpt",
        "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/02_tokenized_datasets/knn_ner_sent",
        dev_dataloader)
