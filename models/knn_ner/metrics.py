#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : metrics.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2021/01/14 16:13
@version: 1.0
@desc  :
"""

import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import torch
from typing import Any, List, Optional
from torchmetrics import Metric
from models.metrics import ASPMetrics


class SpanF1ForNER(Metric):
    """
    compute span-level F1 scores for named entity recognition task.
    """
    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self, entity_labels: List[str]):
        super().__init__()
        self.num_labels = len(entity_labels)
        self.entity_labels = entity_labels
        self.tags2label = {
            label_idx: label_item
            for label_idx, label_item in enumerate(entity_labels)
        }
        self.metrics = ASPMetrics()

    def update(self,
               ids,
               word_maps,
               pred_sequence_labels,
               gold_sequence_labels,
               sequence_mask=None):
        """
        Args:
            pred_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            gold_sequence_labels: torch.LongTensor, shape of [batch_size, sequence_len]
            sequence_mask: Optional[torch.LongTensor], shape of [batch_size, sequence_len].
                        1 for non-[PAD] tokens; 0 for [PAD] tokens
        """
        pred_sequence_labels = pred_sequence_labels.to("cpu").numpy().tolist()
        gold_sequence_labels = gold_sequence_labels.to("cpu").numpy().tolist()
        word_maps = word_maps.to("cpu").numpy().tolist()
        if sequence_mask is not None:
            sequence_mask = sequence_mask.to("cpu").numpy().tolist()
            # [1, 1, 1, 0, 0, 0]

        for item_idx, (doc_id, subword_map, pred_label_item,
                       gold_label_item) in enumerate(
                           zip(ids, word_maps, pred_sequence_labels,
                               gold_sequence_labels)):
            if sequence_mask is not None:
                sequence_mask_item = sequence_mask[item_idx]
                try:
                    token_end_pos = sequence_mask_item.index(
                        0) - 1  # before [PAD] always has an [SEP] token.
                except:
                    token_end_pos = len(sequence_mask_item)
            else:
                token_end_pos = len(gold_label_item)

            pred_label_item = [
                self.tags2label[tmp] for tmp in pred_label_item[:token_end_pos]
            ]
            gold_label_item = [
                self.tags2label[tmp] for tmp in gold_label_item[:token_end_pos]
            ]

            pred_entities = transform_entity_bmes_labels_to_spans(
                pred_label_item)
            gold_entities = transform_entity_bmes_labels_to_spans(
                gold_label_item)

            self.metrics.update(doc_id, pred_entities, gold_entities)

    def compute(self):
        return self.metrics.compute()

    def reset(self):
        super().reset()
        self.metrics.reset()


def count_confusion_matrix(pred_entities, gold_entities):
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_entities:
        if span_item in gold_entities:
            true_positive += 1
            gold_entities.remove(span_item)
        else:
            false_positive += 1
    # these entities are not predicted.
    for span_item in gold_entities:
        false_negative += 1

    return true_positive, false_positive, false_negative


def transform_entity_bmes_labels_to_spans(label_sequence,
                                          classes_to_ignore=None):
    """
    Given a sequence of BMES-{entity type} labels, extracts spans.
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(label_sequence):
        label = label_sequence[index]
        if label[0] == "S":
            spans.append((index, index, label.split("-")[1]))
        elif label[0] == "B":
            sign = 1
            start = index
            start_cate = label.split("-")[1]
            while label[0] != "E":
                index += 1
                if index >= len(label_sequence):
                    spans.append((start, start, start_cate))
                    sign = 0
                    break
                label = label_sequence[index]
                if not (label[0] == "M" or label[0] == "E"):
                    spans.append((start, start, start_cate))
                    sign = 0
                    break
                if label.split("-")[1] != start_cate:
                    spans.append((start, start, start_cate))
                    sign = 0
                    break
            if sign == 1:
                spans.append((start, index, start_cate))
        else:
            if label != "O":
                pass
        index += 1
    return [span for span in spans if span[2] not in classes_to_ignore]


def transform_entity_bio_labels_to_spans(label_sequence,
                                         classes_to_ignore=None):
    """
    Given a sequence of BMES-{entity type} labels, extracts spans.
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(label_sequence):
        label = label_sequence[index]
        if label[0] == "B":
            sign = 1
            start = index
            start_type = label.split("-")[1]
            while label[0] != "O":
                index += 1
                if index >= len(label_sequence):
                    spans.append((start, start, start_type))
                    sign = 0
                    break
                label = label_sequence[index]
                if not label[0] == "I":
                    spans.append((start, index, start_type))
                    sign = 0
                    break
                if label.split("-")[1] != start_type:
                    spans.append((start, start, start_type))
                    sign = 0
                    break
            if sign == 1:
                spans.append((start, index, start_type))
        else:
            if label != "O":
                pass
        index += 1
    return [span for span in spans if span[2] not in classes_to_ignore]


def token_span_to_word_span(token_entities, word_map):
    return [(word_map[start], word_map[end], t)
            for (start, end, t) in token_entities]
