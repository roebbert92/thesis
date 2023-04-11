from collections import defaultdict
from itertools import product
from glob import glob
import os
import json
import pandas as pd
import numpy as np


def create_summary(evaluations_path: str):
    combined_metrics = {}
    for metrics_file in glob(
            os.path.join(evaluations_path, "*/model_metrics.json")):
        experiment_name = os.path.dirname(metrics_file).split(os.path.sep)[-1]
        combined_metrics[experiment_name] = {}
        with open(metrics_file, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        # collect all in a proper dict
        for run_id, run in metrics.items():
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
                avg_score = np.mean(score)
                std_score = np.std(score)
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
