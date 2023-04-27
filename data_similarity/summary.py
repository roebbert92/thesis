import pandas as pd
from itertools import combinations_with_replacement
import json
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from locale import strcoll
from functools import cmp_to_key

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_similarity.cosine import dataset_similarity


def get_similarity_data(dataset_files: dict) -> pd.DataFrame:
    records = []
    for left, right in combinations_with_replacement(dataset_files, 2):
        with open(dataset_files[left], encoding="utf-8") as file:
            left_dataset = json.load(file)
        if left == right:
            similarity = dataset_similarity(left_dataset)
        else:
            with open(dataset_files[right], encoding="utf-8") as file:
                right_dataset = json.load(file)
            similarity = dataset_similarity(left_dataset, right_dataset)
        for data_type, data in similarity.items():
            records.append({
                "first": left,
                "second": right,
                "data_type": data_type,
                "cosine_similarity": data[0]
            })
            records.append({
                "first": right,
                "second": left,
                "data_type": data_type,
                "cosine_similarity": data[1]
            })

    return pd.DataFrame.from_records(records)


def visualize_similarity_data(data: pd.DataFrame, data_type: str):
    conf_matrix = data.loc[data["data_type"] == data_type].pivot_table(
        "cosine_similarity", index="first", columns="second")
    datasets = conf_matrix.columns.to_list()
    cmp_dataset_names = cmp_to_key(compare_dataset_names)
    datasets.sort(key=cmp_dataset_names)
    conf_matrix = conf_matrix[datasets].sort_index(
        level="first", key=lambda x: pd.Series([datasets.index(y) for y in x]))
    sns_plot = sns.heatmap(conf_matrix,
                           annot=True,
                           cmap=sns.color_palette("Blues", as_cmap=True))
    return sns_plot.get_figure()


def compare_dataset_names(a, b):
    scores = {"train": 0, "dev": 1, "test": 2}
    a1, a2 = a.split("_")
    b1, b2 = b.split("_")
    dataset_eq = strcoll(a1, b1)
    if dataset_eq != 0:
        return dataset_eq
    if a2 == b2:
        return 0
    return scores[a2] - scores[b2]


if __name__ == "__main__":
    dataset_files = {
        "wnut_train":
        "/home/loebbert/projects/thesis/data/wnut/wnut_train.json",
        "wnut_dev":
        "/home/loebbert/projects/thesis/data/wnut/wnut_dev.json",
        "wnut_test":
        "/home/loebbert/projects/thesis/data/wnut/wnut_test.json",
        "lowner_train":
        "/home/loebbert/projects/thesis/data/mlowner/en/lowner_train.json",
        "lowner_dev":
        "/home/loebbert/projects/thesis/data/mlowner/en/lowner_dev.json",
        "lowner_test":
        "/home/loebbert/projects/thesis/data/mlowner/en/lowner_test.json",
    }

    wnut_lowner_sim = get_similarity_data(dataset_files)

    wnut_lowner_sim.to_csv(os.path.join(thesis_path, "data_similarity",
                                        "wnut_lowner_sim.csv"),
                           sep=";",
                           decimal=",")

    plot = visualize_similarity_data(wnut_lowner_sim, "sentences")
    plot.suptitle("WNUT + LOWNER sentences similarity")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "wnut_lowner_sentences.png"),
                dpi=150,
                format="png")

    plot = visualize_similarity_data(wnut_lowner_sim, "gazetteers")
    plot.suptitle("WNUT + LOWNER gazetteers similarity")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "wnut_lowner_gazetteers.png"),
                dpi=150,
                format="png")
