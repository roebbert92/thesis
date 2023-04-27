from collections import Counter
import json
from typing import List, Optional


def count_entities(dataset: List[dict]):
    entities = []
    for item in dataset:
        for entity in item["entities"]:
            if entity["end"] - entity["start"] == 0:
                entity["end"] += 1
            entities.append((" ".join(
                item["tokens"][entity["start"]:entity["end"]]).lower(),
                             entity["type"]))
    return Counter(entities)


def overlap_metrics(first: Counter, second: Counter):
    overlap = set(first).intersection(set(second))
    total_count = len(overlap)
    if total_count == 0:
        return 0.0, 0.0
    rel_first = sum([first[item] for item in overlap]) / sum(
        [value for value in first.values()])
    rel_second = sum([second[item] for item in overlap]) / sum(
        [value for value in second.values()])
    return rel_first, rel_second


def dataset_overlap(first: List[dict], second: Optional[List[dict]] = None):
    first_entities = count_entities(first)
    second_entities = count_entities(
        second) if second is not None else first_entities
    return overlap_metrics(first_entities, second_entities)
