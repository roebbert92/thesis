import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from typing import Dict, List
import pandas as pd
from glob import glob
import json
import os
import multiprocessing as mp


def init_process_metrics_file(d, t):
    global datasets
    global types
    datasets = d
    types = t


def process_metrics_file(metrics_filepath):
    from models.metrics import ASPMetrics
    import pickle
    import os
    metrics_list = []
    fp = metrics_filepath.split(os.path.sep)
    with open(metrics_filepath, "rb") as file:
        metrics: ASPMetrics = pickle.load(file)
    dataset = fp[-1][5:].split(".")[0]
    seed = fp[-4][-1]
    model = fp[-2]
    checkpoint = fp[-1][:4]
    for sample_metrics in metrics.metrics_per_sample(
            datasets[dataset], types).to_dict(orient="records"):
        metrics_list.append({
            "seed": seed,
            "model": model,
            "checkpoint": checkpoint,
            "dataset": dataset,
            **sample_metrics
        })
    return metrics_list


def get_per_sample_metrics():
    metrics_file_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "01_performance_per_sample_metrics.pkl")
    if not os.path.exists(metrics_file_path):
        datasets: Dict[str, List[dict]] = {}
        for dataset_name, dataset_path in [
            ("lowner_train",
             os.path.join(thesis_path, "data", "mlowner",
                          "lowner_train.json")),
            ("lowner_dev",
             os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json")),
            ("lowner_test",
             os.path.join(thesis_path, "data", "mlowner", "lowner_test.json")),
        ]:
            with open(dataset_path, "r", encoding="utf-8") as file:
                datasets[dataset_name] = json.load(file)
        with open(os.path.join(thesis_path, "data", "mlowner",
                               "lowner_types.json"),
                  "r",
                  encoding="utf-8") as file:
            types = list(json.load(file)["entities"].keys())

        all_metrics = []
        metrics_files = list(
            glob(os.path.join(thesis_path, "experiments", "01_performance",
                              "data", r"**", "04_metrics", "**", r"*.pkl"),
                 recursive=True))
        with mp.Pool(mp.cpu_count() - 1,
                     initializer=init_process_metrics_file,
                     initargs=(datasets, types)) as pool:
            for metrics_list in pool.map(process_metrics_file, metrics_files):
                all_metrics.extend(metrics_list)

        metrics_df = pd.DataFrame.from_records(all_metrics)
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        metrics_df.to_pickle(metrics_file_path)
    else:
        metrics_df = pd.read_pickle(metrics_file_path)
    return metrics_df


def get_f1(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset"],
        values=["tp", "fp", "fn"],
        aggfunc="sum")
    agg_df["precision"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fp"])
    agg_df["recall"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fn"])
    agg_df["f1"] = 2 * agg_df["precision"] * agg_df["recall"] / (
        agg_df["precision"] + agg_df["recall"])

    avg_f1 = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                values=["f1"],
                                aggfunc=("mean", "std")).reset_index()

    return avg_f1[(avg_f1["checkpoint"] == checkpoint)
                  & (avg_f1["dataset"] == dataset)]


def get_error_types(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    error_types = ["error_type1", "error_type2", "error_type3", "error_type4", "error_type5"]
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset"],
        values=error_types,
        aggfunc="sum")
    avg_errors = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                values=error_types,
                                aggfunc=("mean", "std")).reset_index()

    return avg_errors[(avg_errors["checkpoint"] == checkpoint)
                  & (avg_errors["dataset"] == dataset)]