import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from models.metrics import ASPMetrics
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_augmentation_errors(aug_dataset: List[dict], dataset: List[dict]):
    assert len(aug_dataset) == len(dataset)
    metrics = ASPMetrics()
    spelling_errors = 0

    def to_prediction(entities: List[dict]):
        return [(ent["start"], ent["end"], ent["type"]) for ent in entities]

    for aug_sample, sample in zip(aug_dataset, dataset):
        assert aug_sample["doc_id"] == sample["doc_id"]
        metrics.update(sample["doc_id"], to_prediction(aug_sample["entities"]),
                       to_prediction(sample["entities"]))
        for token_idx in range(len(sample["tokens"])):
            if aug_sample["tokens"][token_idx] != sample["tokens"][token_idx]:
                spelling_errors += 1

    errors = metrics.errors()
    return {
        "spelling_errors": spelling_errors,
        "error_type1": errors[0],
        "error_type2": errors[1],
        "error_type3": errors[2],
        "error_type4": errors[3],
        "error_type5": errors[4],
    }


def visualize_error_types_augmented_dataset(error_df: pd.DataFrame):
    error_pivot = error_df.pivot_table(columns="seed")
    rows = error_pivot.index.to_list()
    categories = error_pivot.columns.to_list()
    column_values = {
        f"seed_{col}": error_pivot[col].to_list()
        for col in categories
    }

    x = np.arange(len(rows)) * max(len(rows) // 3, 1)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')

    for attribute, measurement in column_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('# of errors')
    ax.set_title('Errors by type in augmented dataset')
    ax.set_xticks(x + width, rows)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, max(error_df.max()) + 300)

    plt.show()