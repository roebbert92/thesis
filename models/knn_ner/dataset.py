#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : ner_dataset.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/08 20:09
@version: 1.0
@desc  :
"""

import os

import torch
from torch.utils.data import Dataset
from tokenizers.implementations import BertWordPieceTokenizer
from transformers.models.xlm_roberta import XLMRobertaTokenizer
from transformers.models.roberta import RobertaTokenizer
from transformers.utils import PaddingStrategy


class NERDataset(Dataset):

    __ner_labels = None
    """the Dataset Class for NER task."""
    def __init__(self,
                 directory,
                 entity_labels,
                 prefix,
                 vocab_file,
                 max_length=512,
                 file_name="char.bmes",
                 lower_case=False,
                 language="en",
                 en_roberta=False):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            config_path: str, config_path must contain [pinyin_map.json, id2pinyin.json, pinyin2tensor.json]
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory,
                                      "{}.{}".format(prefix, file_name))
        self.data_items = NERDataset._read_conll(data_file_path)
        self.en_roberta = en_roberta
        #self.tokenizer = XLMRobertaTokenizer.from_pretrained(vocab_file)
        self.tokenizer: XLMRobertaTokenizer = XLMRobertaTokenizer.from_pretrained(
            vocab_file)
        self.label_to_idx = {
            label_item: label_idx
            for label_idx, label_item in enumerate(entity_labels)
        }
        self.idx_to_label = {}
        for key, value in self.label_to_idx.items():
            self.idx_to_label[int(value)] = key
        self.language = language

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence, doc_idx = data_item[0], data_item[
            1], data_item[2]
        label_sequence = [
            self.label_to_idx[label_item] for label_item in label_sequence
        ]

        concate_word = ""
        if (self.language == "en"):
            concate_word = " "
        token_sequence = concate_word.join(
            token_sequence[:min(self.max_length - 2, len(token_sequence))])

        #token_sequence = token_sequence[:self.max_length - 2]
        label_sequence = label_sequence[:min(self.max_length -
                                             2, len(label_sequence))]
        # convert string to ids
        tokenizer_output = [self.tokenizer.encode(x if idx > 0 else " " + x, truncation=True, stride=self.max_length//2, return_overflowing_tokens=True, padding=PaddingStrategy.LONGEST, add_special_tokens=False) for idx, x in enumerate(token_sequence.split(concate_word))]

        
        bert_tokens = [token_id for token_ids in tokenizer_output for token_id in token_ids]
        subword_map = [token_idx for token_idx, token_ids in enumerate(tokenizer_output) for _ in token_ids]

        assert len(bert_tokens) <= self.max_length
        #assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.nn.functional.pad(torch.tensor(bert_tokens, dtype=torch.long), (1, 1), value=self.tokenizer.pad_token_id)
        label = torch.tensor(label_sequence, dtype=torch.long)
        token_map = torch.tensor(subword_map, dtype=torch.long)
        
        return doc_idx, token_map, input_ids, label

    def _update_labels_using_tokenize_offsets(self, offsets,
                                              original_sequence_labels):
        """part of offset sequence [(51, 52), (52, 54)] -> (token index after tokenized, original token index)"""
        update_sequence_labels = []
        for offset_idx, offset_item in enumerate(offsets):
            if offset_idx == 0 or offset_idx == (len(offsets) - 1):
                continue
            update_index, origin_index = offset_item
            current_label = original_sequence_labels[origin_index - 1]
            update_sequence_labels.append(current_label)
        update_sequence_labels = [
            self.label_to_idx["O"]
        ] + update_sequence_labels + [self.label_to_idx["O"]]
        return update_sequence_labels

    def _update_labels_using_tokenize_offsets_english(self, token_data,
                                                      original_sequence_labels,
                                                      token_sequence):
        '''
        vis = {}
        now_idx = 0
        for idx_ in range(len(token_sequence)):
            if (idx_ and token_sequence[idx_-1] == ' '):
                now_idx += 1
            vis[idx_] = now_idx
        update_sequence_labels = [self.label_to_idx["O"]]
        for offset_idx, offset_item in enumerate(token_data.offsets):
            if offset_idx == 0 or offset_idx == (len(token_data.offsets) - 1):
                continue
            update_sequence_labels.append(original_sequence_labels[vis[offset_item[0]]])
        update_sequence_labels.append(self.label_to_idx["O"])
        return update_sequence_labels
        '''
        vis = {}
        now_idx = 0
        for idx_ in range(len(token_sequence)):
            if (idx_ and token_sequence[idx_ - 1] == ' '):
                now_idx += 1
            vis[idx_] = now_idx
        update_sequence_labels = [self.label_to_idx["O"]]
        last = -1
        for offset_idx, offset_item in enumerate(token_data.offsets):
            if offset_idx == 0 or offset_idx == (len(token_data.offsets) - 1):
                continue
            if (vis[offset_item[0]] == last and self.idx_to_label[
                    original_sequence_labels[vis[offset_item[0]]]][0] != 'S'
                    and self.idx_to_label[original_sequence_labels[vis[
                        offset_item[0]]]][0] != 'O'):
                if (self.idx_to_label[original_sequence_labels[vis[
                        offset_item[0]]]][0] == 'E'):
                    update_sequence_labels[-1] = self.label_to_idx[
                        'M' + self.idx_to_label[original_sequence_labels[vis[
                            offset_item[0]]]][1:]]
                    update_sequence_labels.append(
                        original_sequence_labels[vis[offset_item[0]]])
                else:
                    update_sequence_labels.append(self.label_to_idx[
                        'M' + self.idx_to_label[original_sequence_labels[vis[
                            offset_item[0]]]][1:]])
            else:
                update_sequence_labels.append(
                    original_sequence_labels[vis[offset_item[0]]])
            last = vis[offset_item[0]]
        update_sequence_labels.append(self.label_to_idx["O"])
        return update_sequence_labels

    def _bio_roberta_update_labels_using_tokenize_offsets_english(
            self, token_sequence, original_sequence_labels):
        updated_sequence_labels = [self.label_to_idx["O"]]
        for idx_, token_word in enumerate(token_sequence):
            token_idx = self.tokenizer.encode(
                token_word if idx_ == 0 else " " + token_word,
                add_special_tokens=False)
            for sub_idx in range(len(token_idx)):
                if sub_idx > 0 and self.idx_to_label[
                        original_sequence_labels[idx_]][0] == 'B':
                    updated_sequence_labels.append(self.label_to_idx[
                        'I' +
                        self.idx_to_label[original_sequence_labels[idx_]][1:]])
                # elif sub_idx != len(token_idx) - 1 and len(
                #         token_idx) > 1 and self.idx_to_label[
                #             original_sequence_labels[idx_]][0] == 'E':
                #     updated_sequence_labels.append(self.label_to_idx[
                #         'M' +
                #         self.idx_to_label[original_sequence_labels[idx_]][1:]])
                else:
                    updated_sequence_labels.append(
                        original_sequence_labels[idx_])
        updated_sequence_labels.append(self.label_to_idx["O"])
        return updated_sequence_labels

    def _roberta_update_labels_using_tokenize_offsets_english(
            self, token_sequence, original_sequence_labels):
        updated_sequence_labels = [self.label_to_idx["O"]]
        subword_map = []
        for idx_, token_word in enumerate(token_sequence):
            token_idx = self.tokenizer.encode(
                token_word if idx_ == 0 else " " + token_word,
                add_special_tokens=False)
            for sub_idx in range(len(token_idx)):
                subword_map.append(idx_)
                if sub_idx > 0 and self.idx_to_label[
                        original_sequence_labels[idx_]][0] == 'B':
                    updated_sequence_labels.append(self.label_to_idx[
                        'M' +
                        self.idx_to_label[original_sequence_labels[idx_]][1:]])
                elif sub_idx != len(token_idx) - 1 and len(
                        token_idx) > 1 and self.idx_to_label[
                            original_sequence_labels[idx_]][0] == 'E':
                    updated_sequence_labels.append(self.label_to_idx[
                        'M' +
                        self.idx_to_label[original_sequence_labels[idx_]][1:]])
                else:
                    updated_sequence_labels.append(
                        original_sequence_labels[idx_])
        updated_sequence_labels.append(self.label_to_idx["O"])
        return subword_map, updated_sequence_labels

    @classmethod
    def get_labels(cls, types):
        """gets the list of labels for this data set."""

        if (cls.__ner_labels is None):
            cls.__ner_labels = ["O"]
            for t in types:
                for pos in ["B", "I"]:
                    cls.__ner_labels.append("-".join([pos, t]))

        return cls.__ner_labels

    @staticmethod
    def _read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        dataset_name = os.path.basename(os.path.splitext(input_file)[0])

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if len(data_line) == 0:
                if (len(cached_token) != 0 and len(cached_label) != 0):
                    dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[
                    1]
                cached_token.append(token_data_line)
                cached_label.append(label_data_line)
        for idx, item in enumerate(dataset_item_lst):
            item.append(dataset_name + "_" + str(idx))
        return dataset_item_lst
