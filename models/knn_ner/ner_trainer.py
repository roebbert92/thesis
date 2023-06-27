#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : ner_trainer.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/07 16:40
@version: 1.0
@desc  :
"""
import copy
import json
import pickle
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import argparse
from functools import partial
from collections import namedtuple

from models.knn_ner.utils import collate_to_max_length
from models.knn_ner.dataset import NERDataset
from models.knn_ner.metrics import SpanF1ForNER
from models.knn_ner.get_labels import get_labels
from hyperparameter_tuning.utils import factors

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertConfig, BertForTokenClassification
from transformers.models.roberta import RobertaConfig, RobertaForTokenClassification
from transformers.models.xlm_roberta import XLMRobertaConfig, XLMRobertaForTokenClassification

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


@torch.jit.script_if_tracing
def fill_masked_elements(
    all_token_embeddings: torch.Tensor,
    sentence_hidden_states: torch.Tensor,
    mask: torch.Tensor,
    word_ids: torch.Tensor,
    lengths: torch.LongTensor,
):
    for i in torch.arange(int(all_token_embeddings.shape[0])):
        all_token_embeddings[
            i, :lengths[i], :] = insert_missing_embeddings(  # type: ignore
                sentence_hidden_states[i][mask[i] & (word_ids[i] > 0)],
                word_ids[i], lengths[i])
    return all_token_embeddings


@torch.jit.script_if_tracing
def insert_missing_embeddings(token_embeddings: torch.Tensor,
                              word_id: torch.Tensor,
                              length: torch.LongTensor) -> torch.Tensor:
    # in some cases we need to insert zero vectors for tokens without embedding.
    if token_embeddings.shape[0] < length:
        for _id in torch.arange(int(length)):
            if not (word_id == _id).any():
                token_embeddings = torch.cat(
                    (
                        token_embeddings[:_id],
                        torch.zeros_like(token_embeddings[:1]),
                        token_embeddings[_id:],
                    ),
                    dim=0,
                )
    return token_embeddings


class NERTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)  # type: ignore

        self.en_roberta = args.en_roberta
        self.entity_labels = NERDataset.get_labels(args.types)
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)

        # self.bert_config = XLMRobertaConfig.from_pretrained(
        #     self.bert_dir,
        #     output_hidden_states=True,
        #     return_dict=True,
        #     num_labels=self.num_labels,
        #     hidden_dropout_prob=self.args.hidden_dropout_prob)
        # self.model = XLMRobertaForTokenClassification.from_pretrained(
        #     self.bert_dir, config=self.bert_config)
        self.bert_config = XLMRobertaConfig.from_pretrained(
            self.bert_dir,
            output_hidden_states=True,
            return_dict=True,
            num_labels=self.num_labels,
            hidden_dropout_prob=self.args.hidden_dropout_prob)
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            self.bert_dir, config=self.bert_config)

        self.val_metrics = SpanF1ForNER(entity_labels=self.entity_labels)
        self.test_metrics = SpanF1ForNER(entity_labels=self.entity_labels)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight", 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr":
                self.args.lr,
                "weight_decay":
                self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr":
                self.args.lr,
                "weight_decay":
                0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.args.lr,
                                      eps=self.args.adam_epsilon,
                                      weight_decay=self.args.weight_decay,
                                      fused=self.args.fused)

        num_gpus = len(
            [x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) //
                   (self.args.accumulate_grad_batches * num_gpus) +
                   1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def forward(self, input_ids, word_maps, labels):
        attention_mask = (input_ids != 0).long()
        bert_output = self.model.roberta(input_ids,
                                         attention_mask=attention_mask,
                                         output_hidden_states=True)

        loss_mask = torch.cat([
            torch.ones_like(word_maps[:, :1]), (word_maps[:, 1:] != 0).long()
        ],
                              dim=1)
        word_maps[loss_mask == 1] += 1
        word_maps = torch.nn.functional.pad(word_maps, (1, 1), value=0)

        sentence_hidden_states = torch.stack(
            bert_output.hidden_states)[-1, :, :]
        token_lengths = torch.max(word_maps, dim=1).values
        all_token_embeddings = torch.zeros(  # type: ignore
            word_maps.shape[0],
            token_lengths.max(),
            self.bert_config.hidden_size,
            device=self.device)
        no_pad_word_maps = word_maps[:, 1:-1]
        true_tensor = torch.ones_like(no_pad_word_maps[:, :1],
                                      dtype=torch.bool)
        false_tensor = torch.zeros_like(no_pad_word_maps[:, :1],
                                        dtype=torch.bool)
        gain_mask = no_pad_word_maps[:,
                                     1:] != no_pad_word_maps[:, :
                                                             no_pad_word_maps.
                                                             shape[1] - 1]
        first_mask = torch.cat(
            [false_tensor, true_tensor, gain_mask, false_tensor], dim=1)
        all_token_embeddings = fill_masked_elements(all_token_embeddings,
                                                    sentence_hidden_states,
                                                    first_mask, word_maps,
                                                    token_lengths)
        sequence_output = self.model.dropout(all_token_embeddings)
        logits = self.model.classifier(sequence_output)

        loss_mask = torch.zeros(word_maps.shape[0],
                                token_lengths.max(),
                                dtype=torch.long).to(self.device)
        for i in torch.arange(int(loss_mask.shape[0])):
            loss_mask[i, :token_lengths[i]] = 1

        loss = self.compute_loss(logits, labels, loss_mask)

        return TokenClassifierOutput(logits=logits, loss=loss)

    def compute_loss(self, logits, labels, loss_mask=None):
        """
        Desc:
            compute cross entropy loss
        Args:
            logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
            labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
            loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
                1 for non-PAD tokens, 0 for PAD tokens.
        """
        loss_fct = CrossEntropyLoss()
        if loss_mask is not None:
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels))
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        idxs, word_maps, input_ids, labels = batch

        batch_size, seq_len = input_ids.shape
        bert_classifiaction_outputs = self.forward(input_ids=input_ids,
                                                   word_maps=word_maps,
                                                   labels=labels)

        self.log("train_loss",
                 bert_classifiaction_outputs.loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=batch_size)
        return bert_classifiaction_outputs.loss

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()
        super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        idxs, word_maps, input_ids, gold_labels = batch
        batch_size, seq_len = input_ids.shape
        loss_mask = (input_ids != 0).long()
        bert_classification_outputs = self.forward(input_ids=input_ids,
                                                   word_maps=word_maps,
                                                   labels=gold_labels)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(
            bert_classification_outputs.logits)
        self.val_metrics.update(idxs, word_maps, argmax_labels, gold_labels)

    def on_validation_epoch_end(self) -> None:
        errors = self.val_metrics.metrics.errors()
        f1 = self.val_metrics.metrics.f1()
        precision = self.val_metrics.metrics.precision()
        recall = self.val_metrics.metrics.recall()
        self.log_dict(
            {
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
                "val_error_type1": errors[0],
                "val_error_type2": errors[1],
                "val_error_type3": errors[2],
                "val_error_type4": errors[3],
                "val_error_type5": errors[4],
            },
            logger=True,
            on_epoch=True)
        super().on_validation_epoch_end()

    def train_dataloader(self, ) -> DataLoader:
        return self.get_dataloader(self.args.data_prefix + "train")

    def val_dataloader(self, ) -> DataLoader:
        return self.get_dataloader(self.args.data_prefix + "dev")

    def _load_dataset(self, prefix="test"):
        vocab_file = self.args.bert_path
        if not self.en_roberta:
            vocab_file = os.path.join(self.args.bert_path, "vocab.txt")
        dataset = NERDataset(directory=self.args.data_dir,
                             entity_labels=self.entity_labels,
                             prefix=prefix,
                             vocab_file=vocab_file,
                             max_length=self.args.max_length,
                             file_name=self.args.file_name,
                             lower_case=self.args.lower_case,
                             language=self.args.language,
                             en_roberta=self.en_roberta)

        return dataset

    def get_dataloader(self, prefix="train", limit=None) -> DataLoader:
        """return {train/dev/test} dataloader"""
        dataset = self._load_dataset(prefix=prefix)

        if prefix.endswith("train"):
            batch_size = self.args.train_batch_size
            # small dataset like weibo ner, define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.args.eval_batch_size
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(
            dataset=dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            num_workers=3,
            collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
            drop_last=False,
            persistent_workers=False,
            pin_memory=True,
        )

        return dataloader

    def test_dataloader(self, ) -> DataLoader:
        return self.get_dataloader(self.args.data_prefix + "test")

    def val_train_dataloader(self, ) -> DataLoader:
        dataset = self._load_dataset(prefix=self.args.data_prefix + "train")

        batch_size = self.args.eval_batch_size
        data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(
            dataset=dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            num_workers=3,
            collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
            drop_last=False,
            persistent_workers=False,
            pin_memory=True,
        )

        return dataloader

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx):
        idxs, word_maps, input_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        bert_classification_outputs = self.forward(input_ids=input_ids,
                                                   word_maps=word_maps,
                                                   labels=gold_labels)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(
            bert_classification_outputs.logits)
        self.test_metrics.update(idxs, word_maps, argmax_labels, gold_labels)

    def on_test_epoch_end(self) -> None:
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

    def postprocess_logits_to_labels(self, logits):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(
            logits, dim=2)  # shape of [batch_size, seq_len, num_labels]
        argmax_labels = torch.argmax(
            probabilities, 2, keepdim=False)  # shape of [batch_size, seq_len]
        return probabilities, argmax_labels


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers",
                        type=int,
                        default=0,
                        help="num workers for dataloader")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory",
                        action="store_true",
                        help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--save_topk",
                        default=1,
                        type=int,
                        help="save topk checkpoint")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--classifier", type=str, default="single")
    parser.add_argument("--no_lr_scheduler", action="store_true")
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
    parser.add_argument("--lower_case",
                        default=False,
                        type=bool,
                        help="lowercase when load English data.")
    parser.add_argument("--language",
                        default="en",
                        type=str,
                        help="the language of the dataset.")
    parser.add_argument("--en_roberta",
                        action="store_true",
                        help="whether load roberta for classification or not.")

    return parser


def experiment01():
    seeds = [1, 2, 3]
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
            0.01,
            "hidden_dropout_prob":
            0.2,
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
            "bmes",
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
            "knn_ner"
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

                model = NERTask(argparse.Namespace(**train_config))
                # model = NERTask.load_from_checkpoint(
                #     os.path.join(checkpoint_base_path, "last.ckpt"))

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
                trainer.fit(model)
                trainer.save_checkpoint(
                    os.path.join(checkpoint_base_path, "last.ckpt"))

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
                             ckpt_path=checkpoint_best.best_model_path)
                save_metrics("lowner_train", "best")
                trainer.test(model,
                             model.val_dataloader(),
                             ckpt_path=checkpoint_best.best_model_path)
                save_metrics("lowner_dev", "best")
                trainer.test(model,
                             model.test_dataloader(),
                             ckpt_path=checkpoint_best.best_model_path)
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


# def evaluate():
#     parser = get_parser()
#     parser = Trainer.add_argparse_args(parser)
#     args = parser.parse_args()
#
#     model = NERTask.load_from_checkpoint(
#         checkpoint_path=args.checkpoint_path,
#         hparams_file=args.path_to_model_hparams_file,
#         map_location=None,
#         batch_size=1)
#     trainer = Trainer.from_argparse_args(args, deterministic=True)
#
#     trainer.test(model)

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    experiment01()