from collections import Counter
from itertools import product
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def count_entities_dataset(dataset: List[dict]):
    entity_type_count = []
    entity_count = []
    types = set()
    for item in dataset:
        for entity in item["entities"]:
            if entity["end"] - entity["start"] == 0:
                entity["end"] += 1
            entity_text = " ".join(
                item["tokens"][entity["start"]:entity["end"]]).lower()
            entity_type_count.append((entity_text, entity["type"]))
            entity_count.append(entity_text)
            types.add(entity["type"])

    return Counter(entity_count), Counter(entity_type_count), types


def count_entities_gazetteer(gazetteer: List[dict]):
    entity_type_count = []
    entity_count = []
    types = set()
    for item in gazetteer:
        entity_text = item["entity"].lower()
        entity_type_count.append((entity_text, item["type"]))
        entity_count.append(entity_text)
        types.add(item["type"])

    return Counter(entity_count), Counter(entity_type_count), types


def entity_coverage_ratio(first: List[dict],
                          second: List[dict],
                          first_is_dataset=True,
                          second_is_dataset=True):
    if first_is_dataset:
        first_entity_count, first_entity_type_count, first_types = count_entities_dataset(
            first)
    else:
        first_entity_count, first_entity_type_count, first_types = count_entities_gazetteer(
            first)
    if second_is_dataset:
        second_entity_count, second_entity_type_count, second_types = count_entities_dataset(
            second)
    else:
        second_entity_count, second_entity_type_count, second_types = count_entities_gazetteer(
            second)

    types = first_types.union(second_types)
    ratio = {}
    c = {}
    expected = 0.0
    total_second_entity_count = 0
    for entity, second_total_count in second_entity_count.items():
        total_second_entity_count += second_total_count
        if entity not in first_entity_count:
            ratio[entity] = 0.0
            c[entity] = 0
        else:
            ratio[entity] = 0.0
            c[entity] = first_entity_count[entity]
            for t in types:
                if (entity, t) not in first_entity_type_count or (
                        entity, t) not in second_entity_type_count:
                    continue
                ratio[entity] += (first_entity_type_count[(entity, t)] /
                                  first_entity_count[entity]
                                  ) * second_entity_type_count[(entity, t)]
            ratio[entity] /= second_total_count
            expected += ratio[entity] * second_total_count
    expected /= total_second_entity_count
    return ratio, c, expected


def display_cases_entity_coverage_ratio(ratio: dict,
                                        c: dict,
                                        name: str,
                                        ax: Optional[plt.Axes] = None):
    case1 = [key for key, value in ratio.items() if value == 1.0]
    case2_1 = [key for key, value in ratio.items() if 0.5 < value < 1.0]
    case2_2 = [key for key, value in ratio.items() if 0.0 < value <= 0.5]
    case3 = [
        key for key, value in ratio.items() if value == 0.0 and c[key] != 0
    ]
    case4 = [
        key for key, value in ratio.items() if value == 0.0 and c[key] == 0
    ]

    x = ["ρ=1", "ρ ∈ (0.5,1)", "ρ ∈ (0,0.5]", "ρ=0∧C≠0", "ρ=0∧C=0"]
    x_ticks = np.arange(len(x))
    y = [len(case1), len(case2_1), len(case2_2), len(case3), len(case4)]

    if ax is None:
        bar_container = plt.bar(x_ticks, y, align="center")
    else:
        bar_container = ax.bar(x_ticks, y, align="center")
    plt.xticks(x_ticks, x)
    plt.ylabel("# of entities")
    plt.bar_label(bar_container, y)
    plt.title(f"Entity Coverage Histogram {name}")

    return plt.figure() if ax is None else ax.figure


def confusion_matrix_expected_entity_coverage_ratio(datasets: List[List[dict]],
                                                    names: List[str],
                                                    are_datasets: List[bool]):
    assert len(datasets) == len(names) == len(are_datasets)
    results = []
    for first_idx, second_idx in product(range(len(datasets)),
                                         range(len(datasets))):
        _, _, expected = entity_coverage_ratio(datasets[first_idx],
                                               datasets[second_idx],
                                               are_datasets[first_idx],
                                               are_datasets[second_idx])
        results.append({
            "first": names[first_idx],
            "second": names[second_idx],
            "expected_entity_coverage_ratio": round(expected, 4)
        })
    return pd.DataFrame.from_records(results)