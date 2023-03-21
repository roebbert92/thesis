import copy
import json
import os
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def one_hot_ignore_negative(labels, num_classes):
    return F.one_hot(
        torch.where((labels >= 0), labels, num_classes),
        num_classes=num_classes+1
    )[..., :-1].bool()


class Tensorizer:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, mention_start, mention_end, num_label_types):
        self.config = config
        self.tokenizer = tokenizer

        self.num_label_types = num_label_types

        self.mention_start_id = self.tokenizer.convert_tokens_to_ids(
            mention_start)
        self.mention_end_id = self.tokenizer.convert_tokens_to_ids(mention_end)

        # Will be used in evaluation
        self.stored_info = {
            'example': {},  # {doc_key: ...}
            'subtoken_maps': {}  # {doc_key: ...}
        }

    def get_action_labels(
        self, label_ids
    ):
        # replacing natural language tokens with <copy>: action 0
        # <m> with action 1
        # </m> with action 2
        action_labels = torch.where(
            label_ids != self.tokenizer.pad_token_id, label_ids, torch.ones_like(
                label_ids)*(-103)
        )
        action_labels = torch.where(
            action_labels == self.mention_start_id, -2, action_labels
        )
        action_labels = torch.where(
            action_labels == self.mention_end_id, -1, action_labels
        )
        action_labels = torch.where(
            (action_labels != -1) & (action_labels != -2) & (action_labels >= 0),
            -3, action_labels
        )
        action_labels += 3
        return action_labels

    def tensorize(
        self, example, is_training
    ):
        # Keep info to store
        doc_key = example['doc_id']
        self.stored_info['subtoken_maps'][doc_key] = example.get(
            'subtoken_map', None)
        self.stored_info['example'][doc_key] = example

        is_training = torch.tensor(is_training, dtype=torch.bool)

        # Sentences/segments
        sentence = copy.deepcopy(example['sentence'])  # Segments
        input_sentence = copy.deepcopy(example['input_sentence'])  # Segments
        target_sentence = copy.deepcopy(example["target_sentence"])

        ent_type_sequence = copy.deepcopy(example['ent_type_sequence'])
        ent_indices = copy.deepcopy(example['ent_indices'])

        input_ids = self.tokenizer.convert_tokens_to_ids(input_sentence)
        to_copy_ids = self.tokenizer.convert_tokens_to_ids(sentence)
        target_ids = self.tokenizer.convert_tokens_to_ids(target_sentence)
        assert isinstance(input_ids, list)
        assert isinstance(target_ids, list)

        input_len, target_len = len(input_ids), len(target_ids)

        input_mask = [1] * input_len
        target_mask = [1] * target_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        to_copy_ids = torch.tensor(to_copy_ids, dtype=torch.long)

        target_ids = torch.tensor(target_ids, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.long)

        action_labels = self.get_action_labels(target_ids)

        ent_types = torch.tensor(ent_type_sequence, dtype=torch.long)
        ent_indices = torch.tensor(ent_indices, dtype=torch.long)

        is_l = (target_ids == self.mention_start_id)
        l_ent_indices = ent_indices[is_l]

        # (target_len, num_l)
        lr_pair_flag = (l_ent_indices.unsqueeze(0) == ent_indices.unsqueeze(1))
        # (target_len, num_l)
        # (target_len, 1, num_class) == (target_len, num_l, 1) -> (target_len, num_l, num_class)
        lr_pair_flag = one_hot_ignore_negative(
            ent_types, num_classes=self.num_label_types
        ).unsqueeze(1) & lr_pair_flag.unsqueeze(-1)

        # Construct example
        tensor = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "to_copy_ids": to_copy_ids,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "action_labels": action_labels,
            "ent_indices": ent_indices,
            "ent_types": ent_types,
            "lr_pair_flag": lr_pair_flag,
            "is_training": is_training,
        }

        return doc_key, tensor


def ner_collate_fn(batch):
    """
        Collate function for the NER dataloader.
    """
    doc_keys, batch = zip(*batch)
    batch = {k: [sample[k] for sample in batch] for k in batch[0]}
    batch_size = len(batch["input_ids"])

    max_input_len = max([sample.size(0) for sample in batch["input_ids"]])
    max_target_len = max([sample.size(0) for sample in batch["target_ids"]])
    max_sent_id_len = max([sample.size(0) for sample in batch["to_copy_ids"]])

    for k in ["to_copy_ids"]:
        batch[k] = torch.stack([
            F.pad(x, (0, max_sent_id_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)
    for k in ["input_ids", "input_mask", "sentence_idx"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len)
        batch[k] = torch.stack([
            F.pad(x, (0, max_input_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)
    for k in ["target_ids", "target_mask",
              "ent_indices", "ent_types",
              "action_labels", "target_sentence_idx"]:
        # (batch_size, max_target_len)
        if k not in batch:
            continue
        batch[k] = torch.stack([
            F.pad(x, (0, max_target_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)

    max_num_l = max([sample.size(1) for sample in batch["lr_pair_flag"]])

    for k in ["lr_pair_flag"]:
        # (batch_size, max_target_len, max_num_l, num_class)
        if max_num_l > 0:
            batch[k] = torch.stack([
                F.pad(x, (0, 0, 0, max_num_l - x.size(1), 0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros(
                (batch_size, max_target_len, 0), dtype=torch.long)

    batch["is_training"] = torch.tensor(batch["is_training"], dtype=torch.bool)
    return doc_keys, batch


class NERDataProcessor(object):
    def __init__(self, config,
                 tokenizer: PreTrainedTokenizer,
                 mention_start,
                 mention_end,
                 train_file,
                 dev_file,
                 test_file,
                 type_file):
        self.config = config
        
        self.tokenizer = tokenizer
        self.tokenizer_name = os.path.basename(os.path.splitext(self.tokenizer.name_or_path)[0])
        self.dir_name = os.path.dirname(type_file)

        # Get tensorized samples
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
        else:
            # Generate tensorized samples
            with open(type_file, encoding="utf-8") as file:
                labels = json.load(file)['entities']
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config, self.tokenizer, mention_start, mention_end, len(labels))
            suffix = f'{self.tokenizer_name}.jsonlines'
            assert suffix in train_file
            assert suffix in dev_file
            assert suffix in test_file

            paths = {
                'train': train_file,
                'dev': dev_file,
                'test': test_file
            }

            for split, path in paths.items():
                is_training = (split == 'train')
                with open(path, encoding="utf-8") as file:
                    samples = json.load(file)
                tensor_samples = [tensorizer.tensorize(
                    sample, is_training) for sample in samples]

                self.tensor_samples[split] = NERDataset(
                    sorted(
                        [(doc_key, tensor)
                         for doc_key, tensor in tensor_samples],
                        key=lambda x: -x[1]['input_ids'].size(0)
                    )
                )
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            pickle.dump((self.tensor_samples, self.stored_info),
                        open(cache_path, 'wb'))

    def get_tensor_samples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['train'], self.tensor_samples['dev'], self.tensor_samples['test']

    def get_stored_info(self):
        return self.stored_info

    def get_cache_path(self):
        cache_path = os.path.join(
            self.dir_name, f'cached.tensors.{self.tokenizer_name}.bin'
        )
        return cache_path
