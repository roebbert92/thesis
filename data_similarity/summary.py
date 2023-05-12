from typing import List
import pandas as pd
from itertools import combinations_with_replacement, product
import json
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from locale import strcoll
from functools import cmp_to_key
import numpy as np

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_similarity.cosine import dataset_similarity
from data_similarity.exact_match import dataset_overlap


def get_data(dataset_files: dict, already_seen: List[tuple] = []):
    similarities = []
    overlaps = []
    cache = {}
    for left, right in combinations_with_replacement(dataset_files, 2):
        if (left, right) in already_seen:
            continue
        with open(dataset_files[left], encoding="utf-8") as file:
            left_dataset = json.load(file)
        if left == right:
            similarity = dataset_similarity(cache, left, left_dataset)
            overlap = dataset_overlap(left_dataset)
        else:
            with open(dataset_files[right], encoding="utf-8") as file:
                right_dataset = json.load(file)
            similarity = dataset_similarity(cache, left, left_dataset, right,
                                            right_dataset)
            overlap = dataset_overlap(left_dataset, right_dataset)
        for direction, data_type, data_ids, data in similarity:
            first = left if direction == "first" else right
            second = right if direction == "first" else left
            similarities.extend([{
                "first": first,
                "second": second,
                "data_type": data_type,
                "data_id": id,
                "cosine_similarity": d
            } for id, d in zip(data_ids, data)])
        for direction, data in overlap:
            overlaps.append({
                "first": left if direction == "first" else right,
                "second": right if direction == "first" else left,
                "overlap": data
            })

    return pd.DataFrame.from_records(similarities), pd.DataFrame.from_records(
        overlaps)


def visualize_overlap_data(data: pd.DataFrame):
    conf_matrix = data.pivot_table("overlap", index="first", columns="second")
    datasets = conf_matrix.columns.to_list()
    cmp_dataset_names = cmp_to_key(compare_dataset_names)
    datasets.sort(key=cmp_dataset_names)
    conf_matrix = conf_matrix[datasets].sort_index(
        level="first", key=lambda x: pd.Series([datasets.index(y) for y in x]))
    sns_plot = sns.heatmap(conf_matrix,
                           annot=True,
                           cmap=sns.color_palette("Blues", as_cmap=True))
    return sns_plot.get_figure()


def visualize_similarity_data(data: pd.DataFrame, data_type: str):
    conf_matrix = data.loc[data["data_type"] == data_type].pivot_table(
        "cosine_similarity", index="first", columns="second", aggfunc=np.mean)
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


def conll_lowner_wnut():
    dataset_files = {
        "conll03_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/conll03/conll03_train.json",
        "conll03_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/conll03/conll03_dev.json",
        "conll03_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/conll03/conll03_test.json",
        "wnut_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/wnut/wnut_train.json",
        "wnut_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/wnut/wnut_dev.json",
        "wnut_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/wnut/wnut_test.json",
        "lowner_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/mlowner/lowner_train.json",
        "lowner_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/mlowner/lowner_dev.json",
        "lowner_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/mlowner/lowner_test.json",
    }

    if os.path.exists(
            os.path.join(thesis_path, "data_similarity",
                         "conll_wnut_lowner_sim.csv")):
        wnut_lowner_sim = pd.read_csv(os.path.join(thesis_path,
                                                   "data_similarity",
                                                   "conll_wnut_lowner_sim.csv"),
                                      sep=";",
                                      decimal=",").reset_index()
        wnut_lowner_overlap = pd.read_csv(os.path.join(
            thesis_path, "data_similarity", "conll_wnut_lowner_overlap.csv"),
                                          sep=";",
                                          decimal=",").reset_index()

        already_seen = list(
            product(wnut_lowner_overlap["first"].unique(), repeat=2))

        conll_wnut_lowner_sim, conll_wnut_lowner_overlap = get_data(
            dataset_files, already_seen)

        sims = wnut_lowner_sim.to_dict(orient="records")
        sims.extend(conll_wnut_lowner_sim.to_dict("records"))
        wnut_lowner_sim = pd.DataFrame.from_records(sims)

        overlaps = wnut_lowner_overlap.to_dict(orient="records")
        overlaps.extend(conll_wnut_lowner_overlap.to_dict("records"))
        wnut_lowner_overlap = pd.DataFrame.from_records(overlaps)
    else:
        wnut_lowner_sim, wnut_lowner_overlap = get_data(dataset_files)
    wnut_lowner_sim["cosine_similarity"] = wnut_lowner_sim[
        "cosine_similarity"].clip(-1, 1)

    wnut_lowner_sim.to_csv(os.path.join(thesis_path, "data_similarity",
                                        "conll_wnut_lowner_sim.csv"),
                           sep=";",
                           decimal=",", index=False)
    wnut_lowner_overlap.to_csv(os.path.join(thesis_path, "data_similarity",
                                            "conll_wnut_lowner_overlap.csv"),
                               sep=";",
                               decimal=",", index=False)

    plt.figure(figsize=(10, 8))
    plot = visualize_similarity_data(wnut_lowner_sim, "sentences")
    plot.suptitle("WNUT + LOWNER sentences similarity")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "conll_wnut_lowner_sentences.png"),
                dpi=150,
                format="png")

    plt.close()

    plt.figure(figsize=(10, 8))
    plot = visualize_similarity_data(wnut_lowner_sim, "gazetteers")
    plot.suptitle("WNUT + LOWNER gazetteers similarity")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "conll_wnut_lowner_gazetteers.png"),
                dpi=150,
                format="png")

    plt.close()

    plt.figure(figsize=(10, 8))
    plot = visualize_overlap_data(wnut_lowner_overlap)
    plot.suptitle("WNUT + LOWNER entity overlap")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "conll_wnut_lowner_overlap.png"),
                dpi=150,
                format="png")


def cross_ner():
    dataset_files = {
        "conll03_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/conll03/conll03_train.json",
        "conll03_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/conll03/conll03_dev.json",
        "conll03_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/conll03/conll03_test.json",
        "wnut_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/wnut/wnut_train.json",
        "wnut_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/wnut/wnut_dev.json",
        "wnut_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/wnut/wnut_test.json",
        "ai_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/ai/ai_train.json",
        "ai_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/ai/ai_dev.json",
        "ai_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/ai/ai_test.json",
        "literature_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/literature/literature_train.json",
        "literature_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/literature/literature_dev.json",
        "literature_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/literature/literature_test.json",
        "music_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/music/music_train.json",
        "music_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/music/music_dev.json",
        "music_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/music/music_test.json",
        "politics_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/politics/politics_train.json",
        "politics_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/politics/politics_dev.json",
        "politics_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/politics/politics_test.json",
        "science_train":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/science/science_train.json",
        "science_dev":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/science/science_dev.json",
        "science_test":
        "/Users/robinloebbert/Masterarbeit/thesis/data/crossner/science/science_test.json",
    }

    if os.path.exists(
            os.path.join(thesis_path, "data_similarity", "cross_sim.csv")):
        cross_sim = pd.read_csv(os.path.join(thesis_path, "data_similarity",
                                             "cross_sim.csv"),
                                sep=";",
                                decimal=",").reset_index()
        cross_overlap = pd.read_csv(os.path.join(thesis_path,
                                                 "data_similarity",
                                                 "cross_overlap.csv"),
                                    sep=";",
                                    decimal=",").reset_index()
    else:
        cross_sim, cross_overlap = get_data(dataset_files)
    cross_sim["cosine_similarity"] = cross_sim["cosine_similarity"].clip(-1, 1)

    cross_sim.to_csv(os.path.join(thesis_path, "data_similarity",
                                  "cross_sim.csv"),
                     sep=";",
                     decimal=",",
                     index=False)
    cross_overlap.to_csv(os.path.join(thesis_path, "data_similarity",
                                      "cross_overlap.csv"),
                         sep=";",
                         decimal=",",
                         index=False)

    plt.figure(figsize=(15, 15))
    plot = visualize_similarity_data(cross_sim, "sentences")
    plot.suptitle("CrossNER sentences similarity")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "cross_sentences.png"),
                dpi=150,
                format="png")

    plt.close()

    plt.figure(figsize=(15, 15))
    plot = visualize_similarity_data(cross_sim, "gazetteers")
    plot.suptitle("CrossNER gazetteers similarity")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "cross_gazetteers.png"),
                dpi=150,
                format="png")

    plt.close()

    plt.figure(figsize=(15, 15))
    plot = visualize_overlap_data(cross_overlap)
    plot.suptitle("CrossNER entity overlap")
    plt.savefig(os.path.join(thesis_path, "data_similarity",
                             "cross_overlap.png"),
                dpi=150,
                format="png")
