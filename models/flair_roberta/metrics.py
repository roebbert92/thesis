from dataclasses import asdict, dataclass
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

            pred_entities = transform_entity_bio_labels_to_spans(
                pred_label_item)
            gold_entities = transform_entity_bio_labels_to_spans(
                gold_label_item)

            self.metrics.update(doc_id, pred_entities, gold_entities)

    def compute(self):
        return self.metrics.compute()

    def reset(self):
        self.metrics.reset()
        super().reset()

@dataclass()
class Entity():
    type: str
    start: int
    end: int

def transform_entity_bio_labels_to_spans(label_sequence,
                                         classes_to_ignore=None):
    """
    Given a sequence of BMES-{entity type} labels, extracts spans.
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    current_types = {}
    while index < len(label_sequence):
        types = []
        bio_label = label_sequence[index][0]
        type = label_sequence[index][2:]
        types.append(type)
        if bio_label == "B":
            # check if same type is already begun before
            if type in current_types:
                current_types[type].end = index
                assert current_types[type].start < current_types[
                    type].end
                spans.append(asdict(current_types[type]))
            # B = start entity + recognize type
            current_types[type] = Entity(type, index, index)
        elif bio_label == "I":
            # I = current_type.end += 1
            # According to paper there are entities that start with I
            if type in current_types:
                current_types[type].end = index
            else:
                current_types[type] = Entity(type, index, index)
        elif bio_label == "O":
            # O = delete all types from current_types
            for current_type in current_types.values():
                try:
                    current_type.end = index
                    assert current_type.start < current_type.end
                    spans.append(asdict(current_type))
                except AssertionError:
                    print("Annotation error: ", current_type, label_sequence, index)
            current_types = {}
        # if types are not in tags -> entity ended
        types_not_in_tags = set(current_types).difference(set(types))
        for type in types_not_in_tags:
            current_type = current_types[type]
            try:
                current_type.end = index
                assert current_type.start < current_type.end
                spans.append(asdict(current_type))
                del current_types[type]
            except AssertionError:
                print("Annotation error: ",
                        current_type, label_sequence, index)
                
        index += 1
    # clean up current_types
    for current_type in current_types.values():
        try:
            current_type.end = index
            assert current_type.start < current_type.end
            spans.append(asdict(current_type))
        except AssertionError:
            print("Annotation error: ", current_type, label_sequence, index)
    return [(span["start"], span["end"]-1, span["type"]) for span in spans if span["type"] not in classes_to_ignore]


def token_span_to_word_span(token_entities, word_map):
    return [(word_map[start], word_map[end], t)
            for (start, end, t) in token_entities]
