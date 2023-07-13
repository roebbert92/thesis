from collections import defaultdict
import pickle
import sys
import os

import torch

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
from data_metrics.entity_coverage_ratio import calc_ecr_classes, entity_coverage_ratio, entity_coverage_ratio_precounted, count_entities
from data_metrics.search_sample_similarity import SearchSampleSimilarity, get_search_sample_similarity_for_model, SearchSampleSimilarityCollator
from haystack import Document
import shutil


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
    with open(
            datasets[
                f"{seed}_{gazetteer_size}_{error_ratio}_{error_part}_{timestep}_{name[-1]}"],
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
    tmp_filepath = os.path.join(thesis_path, "evaluations", "metrics", "tmp",
                                f"{uuid4()}.parquet")
    os.makedirs(os.path.dirname(tmp_filepath), exist_ok=True)
    metrics_df.to_parquet(tmp_filepath)
    return tmp_filepath


def get_per_sample_metrics():
    metrics_file_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "02_content_per_sample_metrics.parquet")
    if not os.path.exists(metrics_file_path):
        pqwriter = None
        datasets: Dict[str, str] = {}
        datapath = os.path.join(thesis_path, "experiments", "02_content",
                                "data")
        seeds = [1, 2, 3]
        sizes = [2000, 4000, 6000, 8000]
        error_ratios = [0, 5, 10, 15]
        error_parts = ["train", "gazetteer", "both"]
        timesteps = [0, 1, 2]
        parts = ["train", "dev", "test"]

        with open(
                os.path.join(thesis_path, "data", "mlowner",
                             "lowner_types.json")) as file:
            types = list(json.load(file)["entities"].keys())

        for seed, size, error_ratio, error_part, timestep, part in product(
                seeds, sizes, error_ratios, error_parts, timesteps, parts):
            labeled_data_path = os.path.join(datapath, f"seed_{str(seed)}",
                                             "00_datasets", f"size_{size}",
                                             f"error_ratio_{error_ratio}")
            if timestep == 0:
                if error_part in ["train", "both"
                                  ] and part in ["train", "dev"]:
                    datasets[
                        f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                            labeled_data_path, f"error_lowner_{part}.json")
                else:
                    datasets[
                        f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                            thesis_path, "data", "mlowner",
                            f"lowner_{part}.json")

            if timestep > 0:
                datasets[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                        thesis_path, "data", "mlowner", f"lowner_{part}.json")
        # including checkpoint "best" too large for system memory
        metrics_files = [
            fp for fp in
            glob(os.path.join(thesis_path, "experiments", "02_content", "data",
                              r"**", "04_metrics", "**", r"*.pkl"),
                 recursive=True) if os.path.splitext(
                     fp.split(os.path.sep)[-1])[0].split("_")[0] == "last"
        ]
        with mp.Pool(mp.cpu_count() // 2,
                     initializer=init_process_metrics_file,
                     initargs=(datasets, types)) as pool:
            results = pool.map_async(process_metrics_file, metrics_files)
            for filepath in results.get():
                metrics_df = pd.read_parquet(filepath)
                table = pa.Table.from_pandas(metrics_df)
                if pqwriter is None:
                    pqwriter = pq.ParquetWriter(metrics_file_path,
                                                table.schema)
                pqwriter.write_table(table)
        if pqwriter is not None:
            pqwriter.close()
            shutil.rmtree(
                os.path.join(thesis_path, "evaluations", "metrics", "tmp"))

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


def get_labeled_data():
    labeled_data: Dict[str, List[str]] = {}
    datasets: Dict[str, str] = {}
    seeds = [1, 2, 3]
    sizes = [2000, 4000, 6000, 8000]
    error_ratios = [0, 5, 10, 15]
    error_parts = ["train", "gazetteer", "both"]
    parts = ["train", "dev", "test"]
    timesteps = [0, 1, 2]
    datapath = os.path.join(thesis_path, "experiments", "02_content", "data")
    for seed, size, error_ratio, error_part, timestep, part in product(
            seeds, sizes, error_ratios, error_parts, timesteps, parts):
        labeled_data_path = os.path.join(datapath, f"seed_{str(seed)}",
                                         "00_datasets", f"size_{size}",
                                         f"error_ratio_{error_ratio}")
        if timestep == 0:
            if error_part in ["both", "gazetteer"]:
                labeled_data[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = [
                        os.path.join(labeled_data_path,
                                     f"error_sampled_{gaz}.json")
                        for gaz in ["lownergaz", "multiconer"]
                    ]
            else:
                labeled_data[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = [
                        os.path.join(labeled_data_path,
                                     f"clean_sampled_{gaz}.json")
                        for gaz in ["lownergaz", "multiconer"]
                    ]

        if timestep == 1:
            labeled_data[
                f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = [
                    os.path.join(labeled_data_path,
                                 f"clean_sampled_{gaz}.json")
                    for gaz in ["lownergaz", "multiconer"]
                ]

        if timestep == 2:
            labeled_data[
                f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = [
                    os.path.join(thesis_path, "data", "mlowner",
                                 "lowner_gazetteer.json"),
                    os.path.join(thesis_path, "data", "multiconer",
                                 "filtered_multiconer.json")
                ]
        #  process datasets
        datasets_path = os.path.join(datapath, f"seed_{str(seed)}",
                                     "00_datasets", f"size_{size}",
                                     f"error_ratio_{error_ratio}")
        if timestep == 0:
            if error_part in ["train", "both"] and part in ["train", "dev"]:
                datasets[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                        datasets_path, f"error_lowner_{part}.json")
            else:
                datasets[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                        thesis_path, "data", "mlowner", f"lowner_{part}.json")

        if timestep > 0:
            datasets[
                f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                    thesis_path, "data", "mlowner", f"lowner_{part}.json")
    return labeled_data, datasets


def get_labeled_data_entity_coverage():
    eecr_labeled_data_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "02_content_labeled_data_eecr_metrics.parquet")
    if os.path.exists(eecr_labeled_data_data_path):
        eecr_df = pd.read_parquet(eecr_labeled_data_data_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(eecr_labeled_data_data_path),
                    exist_ok=True)
        labeled_data, datasets = get_labeled_data()
        with mp.Pool(mp.cpu_count() - 5) as pool:
            with tqdm(total=len(labeled_data),
                      desc="Compute EECR labeled data") as pbar:
                results = pool.starmap_async(
                    get_ecr_labeled_data_df,
                    [(key, search_result_paths, datasets[key])
                     for key, search_result_paths in labeled_data.items()],
                    callback=lambda _: pbar.update(1))
                for eecr_df in tqdm(results.get(),
                                    total=len(labeled_data),
                                    desc="Concat results to dataframe"):
                    table = pa.Table.from_pandas(eecr_df)
                    if pqwriter is None:
                        pqwriter = pq.ParquetWriter(
                            eecr_labeled_data_data_path, table.schema)
                    pqwriter.write_table(table)
        if pqwriter is not None:
            pqwriter.close()
        eecr_df = pd.read_parquet(eecr_labeled_data_data_path)
    return eecr_df


def get_labeled_data_entity_coverage_per_sample():
    eecr_labeled_data_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "02_content_labeled_data_eecr_per_sample_metrics.parquet")
    if os.path.exists(eecr_labeled_data_data_path):
        eecr_df = pd.read_parquet(eecr_labeled_data_data_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(eecr_labeled_data_data_path),
                    exist_ok=True)
        labeled_data, datasets = get_labeled_data()
        with mp.Pool(mp.cpu_count() - 5) as pool:
            with tqdm(total=len(labeled_data),
                      desc="Compute EECR labeled data") as pbar:
                results = pool.starmap_async(
                    get_ecr_per_sample_labeled_data_df,
                    [(key, search_result_paths, datasets[key])
                     for key, search_result_paths in labeled_data.items()],
                    callback=lambda _: pbar.update(1))
                for eecr_df in tqdm(results.get(),
                                    total=len(labeled_data),
                                    desc="Concat results to dataframe"):
                    table = pa.Table.from_pandas(eecr_df)
                    if pqwriter is None:
                        pqwriter = pq.ParquetWriter(
                            eecr_labeled_data_data_path, table.schema)
                    pqwriter.write_table(table)
        if pqwriter is not None:
            pqwriter.close()
        eecr_df = pd.read_parquet(eecr_labeled_data_data_path)
    return eecr_df


def get_ecr_labeled_data_df(key, labeled_data_paths: List[str], dataset_path):
    seed, size, error_ratio, error_part, timestep, part = key.split("_")
    labeled_data = []
    for labeled_data_path in labeled_data_paths:
        with open(labeled_data_path, "r") as file:
            labeled_data.extend(json.load(file))

    with open(dataset_path, "r") as file:
        dataset = json.load(file)
    eecr_metrics = []
    ratio, c, eecr, _ = entity_coverage_ratio(labeled_data, dataset)
    ecr_classes = calc_ecr_classes(ratio, c)
    eecr_metrics.append({
        "seed": int(seed),
        "gazetteer_size": int(size),
        "error_ratio": int(error_ratio),
        "error_part": error_part,
        "timestep": int(timestep),
        "dataset": "lowner_" + part,
        "eecr": eecr,
        **{key: len(value)
           for key, value in ecr_classes.items()}
    })
    eecr_df = pd.DataFrame.from_records(eecr_metrics)
    return eecr_df


def get_ecr_per_sample_labeled_data_df(key, labeled_data_paths: List[str],
                                       dataset_path):
    seed, size, error_ratio, error_part, timestep, part = key.split("_")
    labeled_data = []
    for labeled_data_path in labeled_data_paths:
        with open(labeled_data_path, "r") as file:
            labeled_data.extend(json.load(file))

    with open(dataset_path, "r") as file:
        dataset = json.load(file)
    eecr_metrics = []
    labeled_data_counted = count_entities(labeled_data)
    for sample in dataset:
        sample_counted = count_entities([sample])
        ratio, c, eecr, _ = entity_coverage_ratio_precounted(
            labeled_data_counted, sample_counted)
        ecr_classes = calc_ecr_classes(ratio, c)
        eecr_metrics.append({
            "seed": int(seed),
            "gazetteer_size": int(size),
            "error_ratio": int(error_ratio),
            "error_part": error_part,
            "timestep": int(timestep),
            "dataset": "lowner_" + part,
            "doc_id": sample["doc_id"],
            "targets": len(sample["entities"]),
            "eecr": eecr,
            **{key: len(value)
               for key, value in ecr_classes.items()}
        })
    eecr_df = pd.DataFrame.from_records(eecr_metrics)
    return eecr_df


def get_search_results_data():
    search_results: Dict[str, str] = {}
    datasets: Dict[str, str] = {}
    seeds = [1, 2, 3]
    sizes = [2000, 4000, 6000, 8000]
    error_ratios = [0, 5, 10, 15]
    error_parts = ["train", "gazetteer", "both"]
    parts = ["train", "dev", "test"]
    timesteps = [0, 1, 2]
    datapath = os.path.join(thesis_path, "experiments", "02_content", "data")
    for seed, size, error_ratio, error_part, timestep, part in product(
            seeds, sizes, error_ratios, error_parts, timesteps, parts):
        search_base_path = os.path.join(datapath, f"seed_{str(seed)}",
                                        "01_search_results", f"size_{size}",
                                        f"error_ratio_{error_ratio}")
        if timestep == 0:
            if error_part in ["both", "gazetteer"]:
                if error_part == "both" and part in ["train", "dev"]:
                    search_results[
                        f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                            search_base_path,
                            f"error_search_results_error_{part}.pkl")
                else:
                    search_results[
                        f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                            search_base_path,
                            f"error_search_results_{part}.pkl")
            else:
                if part in ["train", "dev"]:
                    search_results[
                        f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                            search_base_path,
                            f"sampled_search_results_error_{part}.pkl")
                else:
                    search_results[
                        f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                            search_base_path,
                            f"sampled_search_results_{part}.pkl")

        if timestep == 1:
            search_results[
                f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                    search_base_path, f"sampled_search_results_{part}.pkl")

        if timestep == 2:
            search_results[
                f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                    datapath, "01_search_results",
                    f"search_results_{part}.pkl")
        #  process labeled data
        labeled_data_path = os.path.join(datapath, f"seed_{str(seed)}",
                                         "00_datasets", f"size_{size}",
                                         f"error_ratio_{error_ratio}")
        if timestep == 0:
            if error_part in ["train", "both"] and part in ["train", "dev"]:
                datasets[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                        labeled_data_path, f"error_lowner_{part}.json")
            else:
                datasets[
                    f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                        thesis_path, "data", "mlowner", f"lowner_{part}.json")

        if timestep > 0:
            datasets[
                f"{seed}_{size}_{error_ratio}_{error_part}_{timestep}_{part}"] = os.path.join(
                    thesis_path, "data", "mlowner", f"lowner_{part}.json")
    return search_results, datasets


def get_ecr_per_search_sample_df(key, search_result_path, labeled_data_path):
    seed, size, error_ratio, error_part, timestep, part = key.split("_")
    search = {}
    with open(search_result_path, "rb") as file:
        for idx, items in pickle.load(file).items():
            items: List[Document]
            search[idx] = [item.to_dict() for item in items]
    with open(labeled_data_path, "r") as file:
        dataset = json.load(file)
    eecr_metrics = []

    for idx, sample in enumerate(dataset):
        ratio, c, eecr, _ = entity_coverage_ratio(search[idx], [sample])
        ecr_classes = calc_ecr_classes(ratio, c)
        eecr_metrics.append({
            "seed": int(seed),
            "gazetteer_size": int(size),
            "error_ratio": int(error_ratio),
            "error_part": error_part,
            "timestep": int(timestep),
            "dataset": "lowner_" + part,
            "doc_id": sample["doc_id"],
            "targets": len(sample["entities"]),
            "eecr": eecr,
            **{key: len(value)
               for key, value in ecr_classes.items()}
        })
    eecr_df = pd.DataFrame.from_records(eecr_metrics)
    return eecr_df


def get_search_results_entity_coverage_per_sample():
    eecr_search_results_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "02_content_search_results_eecr_metrics.parquet")
    if os.path.exists(eecr_search_results_data_path):
        eecr_df = pd.read_parquet(eecr_search_results_data_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(eecr_search_results_data_path),
                    exist_ok=True)
        search_results, datasets = get_search_results_data()
        with mp.Pool(mp.cpu_count() - 1) as pool:
            with tqdm(total=len(search_results),
                      desc="Compute EECR search results") as pbar:
                results = pool.starmap_async(
                    get_ecr_per_search_sample_df,
                    [(key, search_result_path, datasets[key])
                     for key, search_result_path in search_results.items()],
                    callback=lambda _: pbar.update(1))
                for eecr_df in tqdm(results.get(),
                                    total=len(search_results),
                                    desc="Concat results to dataframe"):
                    table = pa.Table.from_pandas(eecr_df)
                    if pqwriter is None:
                        pqwriter = pq.ParquetWriter(
                            eecr_search_results_data_path, table.schema)
                    pqwriter.write_table(table)
        if pqwriter is not None:
            pqwriter.close()
        eecr_df = pd.read_parquet(eecr_search_results_data_path)
    return eecr_df


def get_ccr_init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-mpnet-base-v2" if device == "cuda" else "sentence-transformers/all-MiniLM-L6-v2"
    global model
    model = SearchSampleSimilarity(model_name).to(device)
    global collator
    collator = SearchSampleSimilarityCollator(model_name)


def get_ccr_per_sample_df(search_result_path, labeled_data_path):
    search = {}
    with open(search_result_path, "rb") as file:
        for idx, items in pickle.load(file).items():
            items: List[Document]
            search[idx] = [item.to_dict() for item in items]
    with open(labeled_data_path, "r") as file:
        dataset = json.load(file)
    ccr_metrics = []

    for ccr in get_search_sample_similarity_for_model(model, collator, dataset,
                                                      search):
        ccr_metrics.append(ccr)
    return ccr_metrics, (search_result_path, labeled_data_path)


def get_search_results_data_ccr_metrics():
    metrics_file_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "02_content_search_results_ccr_metrics.parquet")
    if os.path.exists(metrics_file_path):
        ccr_metrics_df = pd.read_parquet(metrics_file_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        search_results, datasets = get_search_results_data()
        search_result_combs = defaultdict(list)
        for key, search_result in search_results.items():
            search_result_combs[(search_result, datasets[key])].append(key)
        with mp.get_context("spawn").Pool(1, get_ccr_init) as pool:
            global pbar
            with tqdm(total=len(search_result_combs),
                      desc="CCR search results") as pbar:
                results = pool.starmap_async(get_ccr_per_sample_df,
                                             list(search_result_combs.keys()),
                                             callback=lambda _: pbar.update(1))
                # for key, search_result_path, labeled_data_path in [
                #     (key, search_result_path, labeled_data[key])
                #         for key, search_result_path in search_results.items()
                # ]:
                #     ccr_df = get_ccr_per_sample_df(key, search_result_path,
                #                                labeled_data_path)
                for ccr_result_list, params in tqdm(
                        results.get(),
                        total=len(search_results),
                        desc="Concat results to dataframe"):
                    for key in search_result_combs[params]:
                        seed, size, error_ratio, error_part, timestep, part = key.split(
                            "_")
                        ccr_metrics = []
                        for ccr in ccr_result_list:
                            ccr_metrics.append({
                                "seed": int(seed),
                                "gazetteer_size": int(size),
                                "error_ratio": int(error_ratio),
                                "error_part": error_part,
                                "timestep": int(timestep),
                                "dataset": "lowner_" + part,
                                **ccr
                            })
                        ccr_df = pd.DataFrame.from_records(ccr_metrics)
                        table = pa.Table.from_pandas(ccr_df)
                        if pqwriter is None:
                            pqwriter = pq.ParquetWriter(
                                metrics_file_path, table.schema)
                        pqwriter.write_table(table)
                        pbar.update(1)
        if pqwriter is not None:
            pqwriter.close()

        ccr_metrics_df = pd.read_parquet(metrics_file_path)
    return ccr_metrics_df


if __name__ == "__main__":
    eecr_labeled_data_df = get_labeled_data_entity_coverage()
    eecr_labeled_data_per_sample_df = get_labeled_data_entity_coverage_per_sample(
    )
    eecr_search_results_df = get_search_results_entity_coverage_per_sample()
    ccr_df = get_search_results_data_ccr_metrics()
