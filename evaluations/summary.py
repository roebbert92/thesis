from collections import defaultdict
from itertools import product
from glob import glob
import os
import json
import pandas as pd
import numpy as np


def get_combined_stats(path: str):
    for metrics_file in glob(path):
        experiment_name = os.path.dirname(metrics_file).split(os.path.sep)[-1]
        with open(metrics_file, "r", encoding="utf-8") as file:
            metrics = json.load(file)
            yield metrics, experiment_name


def get_avg_std_from_list(scores: list):
    score = np.asarray(scores)
    avg_score = np.mean(score)
    std_score = np.std(score)
    return avg_score, std_score


def create_summary(evaluations_path: str):
    combined_metrics = {}
    for (metrics, experiment_name) in get_combined_stats(
            os.path.join(evaluations_path, "*/model_metrics.json")):
        combined_metrics[experiment_name] = {}

        # collect all in a proper dict
        for run in metrics.values():
            for model, scores in run.items():
                if model not in combined_metrics[experiment_name]:
                    combined_metrics[experiment_name][model] = defaultdict(
                        list)
                for score_name, score in scores.items():
                    combined_metrics[experiment_name][model][
                        score_name].append(score)

    # Avg + Med + max. abs. diff for both
    flattened_summary = defaultdict(dict)
    for experiment_name, experiment in combined_metrics.items():
        for model, scores in experiment.items():
            for score_name in scores:
                score = np.asarray(scores[score_name])
                avg_score, std_score = get_avg_std_from_list(
                    scores[score_name])
                flattened_summary[(model, score_name)].update(
                    {experiment_name: (avg_score, std_score)})

    index = []
    columns = list(product(combined_metrics.keys(), ["avg", "std"]))

    data = []

    for idx in flattened_summary:
        index.append(idx)
        array = np.asarray([
            item for experiment_name in combined_metrics
            for item in flattened_summary[idx][experiment_name]
        ])
        data.append(array)

    data = np.asarray(data)

    df = pd.DataFrame.from_records(data,
                                   index=pd.MultiIndex.from_tuples(index),
                                   columns=pd.MultiIndex.from_tuples(columns))

    idx = pd.IndexSlice
    dev_avg_df = df.loc[idx[:, "dev_f1"], idx[:, "avg"]].copy()
    dev_avg_df_sorted = dev_avg_df.sort_index(axis=1, level=0)
    dev_avg_df_sorted.to_csv(os.path.join(evaluations_path, "dev_avg_f1.csv"))

    idx = pd.IndexSlice
    dev_std_df = df.loc[idx[:, "dev_f1"], idx[:, "std"]].copy()
    dev_std_df_sorted = dev_std_df.sort_index(axis=1, level=0)
    dev_std_df_sorted.to_csv(os.path.join(evaluations_path, "dev_std_f1.csv"))

    return df, dev_avg_df_sorted, dev_std_df_sorted


def create_database_stats(evaluations_path: str):
    set_intersections = defaultdict(list)
    similarities = defaultdict(list)
    for (metrics, experiment_name) in get_combined_stats(
            os.path.join(evaluations_path, "*/database_metrics.json")):

        for run in metrics.values():
            # set intersections
            for model, overlaps in run["set_intersection"].items():
                for overlap_name, overlap in overlaps.items():
                    set_intersections[(experiment_name, model, overlap_name,
                                       "overlap_count")].append(overlap[0])
                    set_intersections[(experiment_name, model, overlap_name,
                                       "rel_left")].append(overlap[1])
                    set_intersections[(experiment_name, model, overlap_name,
                                       "rel_right")].append(overlap[2])
            # similarities
            for db_name, db_similarities in run["similarity"].items():
                for dataset_name, dataset_entries in db_similarities.items():
                    for entry_name, search_scores in dataset_entries.items():
                        for idx, score in enumerate(search_scores):
                            similarities[(experiment_name, db_name,
                                          dataset_name, entry_name,
                                          idx)].append(score)

    flattened_set_intersections = []
    flattened_similarities = []

    for idx, scores in set_intersections.items():
        avg_score, std_score = get_avg_std_from_list(scores)
        flattened_set_intersections.append({
            "experiment_name": idx[0],
            "model": idx[1],
            "overlap_name": idx[2],
            "metric": idx[3],
            "avg": avg_score,
            "std": std_score
        })

    for idx, scores in similarities.items():
        avg_score, std_score = get_avg_std_from_list(scores)
        flattened_similarities.append({
            "experiment_name": idx[0],
            "db_name": idx[1],
            "dataset_name": idx[2],
            "entry_name": idx[3],
            "result_idx": idx[4],
            "avg": avg_score,
            "std": std_score
        })

    return pd.DataFrame.from_records(
        flattened_set_intersections), pd.DataFrame.from_records(
            flattened_similarities)
