import sys
import os
from typing import Dict, List, Optional

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2]
)
sys.path.append(thesis_path)

import torch
from torch.utils.data import Dataset
from transformers.models.xlm_roberta import XLMRobertaTokenizer
import json
from haystack import Document
import pickle

from data_preprocessing.tokenize import (
    handle_results,
    MENTION_START,
    MENTION_END,
    get_subtokens,
)
from data_preprocessing.tensorize import indices


# TODO: Read normal json dataset; transform to dataset format
class BIONERDataset(Dataset):
    __ner_labels = None

    def __init__(
        self,
        entity_labels,
        directory,
        file_name,
        plm_name,
        max_length=512,
        search_results_dir: Optional[str] = None,
        sent_use_labels: bool = False,
        sent_use_mentions: bool = False,
        gaz_use_labels: bool = False,
        gaz_use_mentions: bool = False,
    ):
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, file_name + ".json")
        self.data_items = BIONERDataset._read_json(data_file_path)
        self.tokenizer: XLMRobertaTokenizer = XLMRobertaTokenizer.from_pretrained(
            plm_name
        )
        self.label_to_idx = {
            label_item: label_idx for label_idx, label_item in enumerate(entity_labels)
        }
        self.idx_to_label = {}
        for key, value in self.label_to_idx.items():
            self.idx_to_label[int(value)] = key

        search_results = {}
        if search_results_dir is not None:
            # context extension with search results
            if MENTION_START not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens(MENTION_START)
            if MENTION_END not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens(MENTION_END)
            with open(
                os.path.join(search_results_dir, file_name + ".pkl"), "rb"
            ) as file:
                search_results: Dict[int, List[Document]] = pickle.load(file)
        self.search_results: Dict[int, List[Document]] = search_results
        self.sent_use_labels = sent_use_labels
        self.sent_use_mentions = sent_use_mentions
        self.gaz_use_labels = gaz_use_labels
        self.gaz_use_mentions = gaz_use_mentions

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        search_result: List[Document] = []
        if idx in self.search_results:
            search_result = self.search_results[idx]
        token_sequence, label_sequence, doc_id = (
            data_item["tokens"],
            data_item["label_sequence"],
            data_item["doc_id"],
        )

        # convert string to ids
        input_tokens = []
        subword_map = []
        label_ids = []
        b_removed = False
        for label_idx, label in enumerate(label_sequence):
            subtokens = get_subtokens(self.tokenizer, token_sequence[label_idx])
            if len(subtokens) == 0:
                # tokens removed
                if label[0] == "B":
                    b_removed = True
            else:
                # append subtokens
                idx = len(input_tokens)
                input_tokens.append(subtokens)
                subword_map.extend([idx for _ in range(len(subtokens))])
                if b_removed:
                    label_ids.append(self.label_to_idx["B" + label[1:]])
                    b_removed = False
                else:
                    label_ids.append(self.label_to_idx[label])

        # append search results with [EOS] token
        processed_input = (
            [self.tokenizer.cls_token]
            + [token for tokens in input_tokens for token in tokens]
            + [self.tokenizer.eos_token]
        )
        handle_results(
            self.tokenizer,
            processed_input,
            search_result,
            self.sent_use_labels,
            self.sent_use_mentions,
            self.gaz_use_labels,
            self.gaz_use_mentions,
        )

        assert len(processed_input) <= self.max_length  # type: ignore
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(processed_input), dtype=torch.long
        )
        labels = torch.tensor(label_ids, dtype=torch.long)
        word_map = torch.tensor(subword_map, dtype=torch.long)
        # input eos -> after pass through BERT model, remove search results tokens
        input_eos = indices(processed_input, self.tokenizer.eos_token)

        return doc_id, word_map, input_ids, labels, input_eos

    @classmethod
    def get_labels(cls, types):
        """gets the list of labels for this data set."""
        if cls.__ner_labels is None:
            cls.__ner_labels = ["O"]
            for t in types:
                for pos in ["B", "I"]:
                    cls.__ner_labels.append("-".join([pos, t]))

        return cls.__ner_labels

    @staticmethod
    def _read_json(input_file):
        with open(input_file, "r", encoding="utf-8") as file:
            items = json.load(file)
        for item in items:
            label_sequence = []
            entity_range = {
                (ent["start"] + span): ent_idx
                for ent_idx, ent in enumerate(item["entities"])
                for span in range(ent["end"] - ent["start"])
            }
            for word_idx in range(len(item["tokens"])):
                if word_idx in entity_range:
                    ent = item["entities"][entity_range[word_idx]]
                    pos = "B"
                    if word_idx > ent["start"]:
                        pos = "I"
                    label_sequence.append(f"{pos}-{ent['type']}")
                else:
                    label_sequence.append("O")
            item["label_sequence"] = label_sequence
        return items
