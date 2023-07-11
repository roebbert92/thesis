import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from typing import Dict, List
import json
import pandas as pd
from glob import glob
from itertools import product
import os
import multiprocessing as mp
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


def init_process_metrics_file(d, t):
    global datasets
    global types
    datasets = d
    types = t


def process_metrics_file(metrics_filepath):
    from models.metrics import ASPMetrics
    import pickle
    import os
    import json
    import pandas as pd
    from uuid import uuid4

    metrics_list = []
    fp = metrics_filepath.split(os.path.sep)
    with open(metrics_filepath, "rb") as file:
        metrics: ASPMetrics = pickle.load(file)
    name = os.path.splitext(fp[-1])[0].split("_")
    timestep = 0
    if name[1] == "sampled":
        timestep = 1
    elif name[1] == "full":
        timestep = 2
    seed = int(fp[-6][-1])
    gazetteer_size = int(fp[-4].split("_")[1])
    error_ratio = int(fp[-3].split("_")[2])
    error_part = fp[-2].split("_")[2]
    with open(datasets[f"{seed}_{gazetteer_size}_{error_ratio}_{name[-1]}"],
              "r") as file:
        dataset = json.load(file)
    metrics_per_sample = metrics.metrics_per_sample(
        dataset, types).to_dict(orient="records")
    for sample_metrics in metrics_per_sample:
        metrics_list.append({
            "seed": seed,
            "checkpoint": name[0],
            "gazetteer_size": gazetteer_size,
            "error_ratio": error_ratio,
            "error_part": error_part,
            "timestep": timestep,
            "dataset": "lowner_" + name[-1],
            **sample_metrics
        })
    if error_ratio == 0 and error_part == "both":
        for part in ["train", "gazetteer"]:
            for sample_metrics in metrics_per_sample:
                metrics_list.append({
                    "seed": seed,
                    "checkpoint": name[0],
                    "gazetteer_size": gazetteer_size,
                    "error_ratio": error_ratio,
                    "error_part": part,
                    "timestep": timestep,
                    "dataset": "lowner_" + name[-1],
                    **sample_metrics
                })
    # save to file
    metrics_df = pd.DataFrame.from_records(metrics_list)
    metrics_path = os.path.join(thesis_path, "evaluations", "metrics", "tmp",
                                str(uuid4()) + ".pkl.tar.gz")
    metrics_df.to_pickle(metrics_path)
    return metrics_path


def metric_files_to_df(metrics_file_path, files: List[str]) -> None:
    pqwriter = None
    # create a parquet write object giving it an output file
    for file in tqdm(files):
        df = pd.read_pickle(file)
        is_last = "last" in list(df["checkpoint"].unique())
        if not is_last:
            continue
        table = pa.Table.from_pandas(df)
        if pqwriter is None:
            pqwriter = pq.ParquetWriter(metrics_file_path, table.schema)
        pqwriter.write_table(table)
    if pqwriter is not None:
        pqwriter.close()


def get_per_sample_metrics():
    metrics_file_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "02_content_per_sample_metrics.parquet")
    if not os.path.exists(metrics_file_path):
        datasets: Dict[str, str] = {}
        seeds = [1, 2, 3]
        sizes = [2000, 4000, 6000, 8000]
        error_ratios = [0, 5, 10, 15]
        parts = ["train", "dev", "test"]

        with open(
                os.path.join(thesis_path, "data", "mlowner",
                             "lowner_types.json")) as file:
            types = list(json.load(file)["entities"].keys())

        for seed, size, error_ratio, part in product(seeds, sizes,
                                                     error_ratios, parts):
            if part in ["train", "dev"]:
                datasets[f"{seed}_{size}_{error_ratio}_{part}"] = os.path.join(
                    thesis_path, "experiments", "02_content", "data",
                    f"seed_{seed}", "00_datasets", f"size_{size}",
                    f"error_ratio_{error_ratio}", f"error_lowner_{part}.json")
            else:
                datasets[f"{seed}_{size}_{error_ratio}_{part}"] = os.path.join(
                    thesis_path, "data", "mlowner", "lowner_test.json")
        metrics_paths = list(
            glob(os.path.join(thesis_path, "evaluations", "metrics", "tmp",
                              r"*.pkl.tar.gz"),
                 recursive=True))
        if len(metrics_paths) == 0:
            metrics_files = reversed(
                list(
                    glob(os.path.join(thesis_path, "experiments", "02_content",
                                      "data", r"**", "04_metrics", "**",
                                      r"*.pkl"),
                         recursive=True)))
            with mp.Pool(mp.cpu_count() - 1,
                         initializer=init_process_metrics_file,
                         initargs=(datasets, types)) as pool:
                for metrics_path in pool.map(process_metrics_file,
                                             metrics_files):
                    if metrics_path is not None:
                        metrics_paths.append(metrics_path)

        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        metric_files_to_df(metrics_file_path, metrics_paths)
    metrics_df = pd.read_parquet(metrics_file_path)
    return metrics_df


def aggregate_performance_metrics(metrics_df: pd.DataFrame):
    agg_df = metrics_df.pivot_table(index=[
        "seed", "gazetteer_size", "error_ratio", "error_part", "timestep",
        "dataset"
    ],
                                    values=["tp", "fp", "fn"],
                                    aggfunc="sum")
    agg_df["precision"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fp"])
    agg_df["recall"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fn"])
    agg_df["f1"] = 2 * agg_df["precision"] * agg_df["recall"] / (
        agg_df["precision"] + agg_df["recall"])

    return agg_df
