from collections import defaultdict
from glob import glob
from itertools import product
import sys
import os
from typing import Dict, List, Tuple

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import pickle
import multiprocessing as mp
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import json
import pandas as pd
from data_metrics.entity_coverage_ratio import calc_ecr_classes, entity_coverage_ratio, entity_coverage_ratio_precounted, count_entities
from data_metrics.search_sample_similarity import SearchSampleSimilarity, get_search_sample_similarity_for_model, SearchSampleSimilarityCollator
from haystack import Document


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

    metrics_list = []
    fp = metrics_filepath.split(os.path.sep)
    with open(metrics_filepath, "rb") as file:
        metrics: ASPMetrics = pickle.load(file)
    name = os.path.splitext(fp[-1])[0].split("_")
    seed = int(fp[-5][-1])
    dims = fp[-3].split("_")
    has_gazetteer = dims[0] == "True"
    finetuning = dims[1]
    pretrained = dims[2] == "True"
    model = "_".join(dims[3:])
    eval_point = fp[-2].split("_")
    gazetteer_content = "None"
    if int(eval_point[0]) == 0:
        gazetteer_content = "lownergaz_sent"
    if int(eval_point[0]) == 1:
        gazetteer_content = "wnut_train"
    if int(eval_point[0]) == 2:
        gazetteer_content = "lownergaz_sent+wnut_train"
    timestep = int(eval_point[1])
    with open(datasets[name[-1]], "r") as file:
        dataset = json.load(file)
    metrics_per_sample = metrics.metrics_per_sample(
        dataset, types).to_dict(orient="records")
    for sample_metrics in metrics_per_sample:
        metrics_list.append({
            "seed": seed,
            "checkpoint": name[0],
            "has_gazetteer": has_gazetteer,
            "finetuning": finetuning,
            "pretrained": pretrained,
            "gazetteer_content": gazetteer_content,
            "timestep": timestep,
            "model": model,
            "dataset": "wnut_" + name[-1],
            **sample_metrics
        })
    metrics_df = pd.DataFrame.from_records(metrics_list)
    return metrics_df


def get_per_sample_metrics():
    metrics_file_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "03_emerging_per_sample_metrics.parquet")
    if not os.path.exists(metrics_file_path):
        pqwriter = None
        datasets: Dict[str, str] = {
            "train": os.path.join(thesis_path, "data", "wnut",
                                  "wnut_train.json"),
            "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
            "test": os.path.join(thesis_path, "data", "wnut",
                                 "wnut_test.json"),
        }

        with open(os.path.join(thesis_path, "data", "wnut",
                               "wnut_types.json")) as file:
            types = list(json.load(file)["entities"].keys())

        metrics_files = list(
            glob(os.path.join(thesis_path, "experiments",
                              "03_adaptation_emerging_entities", "data", r"**",
                              "04_metrics", "**", r"*.pkl"),
                 recursive=True))
        with mp.Pool(mp.cpu_count() - 5,
                     initializer=init_process_metrics_file,
                     initargs=(datasets, types)) as pool:
            results = pool.map_async(process_metrics_file, metrics_files)
            for metrics_df in results.get():
                table = pa.Table.from_pandas(metrics_df)
                if pqwriter is None:
                    pqwriter = pq.ParquetWriter(metrics_file_path,
                                                table.schema)
                pqwriter.write_table(table)
        if pqwriter is not None:
            pqwriter.close()

    metrics_df = pd.read_parquet(metrics_file_path)
    return metrics_df


def get_labeled_data():
    labeled_data_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "03_emerging_labeled_data.pkl.tar.gz")
    gazetteer_comb = [
        [
            ("lownergaz_sent", ),
            ("lownergaz_sent", "wnut_train"),
            ("lownergaz_sent", "wnut_train", "wnut_dev"),
            ("lownergaz_sent", "wnut_train", "wnut_dev", "wnut_test"),
        ],
        [
            ("wnut_train", ),
            ("wnut_train", ),
            ("wnut_train", "wnut_dev"),
            ("wnut_train", "wnut_dev", "wnut_test"),
        ],
        [
            ("lownergaz_sent", "wnut_train"),
            ("lownergaz_sent", "wnut_train"),
            ("lownergaz_sent", "wnut_train", "wnut_dev"),
            ("lownergaz_sent", "wnut_train", "wnut_dev", "wnut_test"),
        ],
    ]
    combination_to_content = defaultdict(list)
    for i, timesteps in enumerate(gazetteer_comb):
        for j, comb in enumerate(timesteps):
            combination_to_content[comb].append((i, j))

    if os.path.exists(labeled_data_path):
        with open(labeled_data_path, "rb") as file:
            labeled_data, datasets = pickle.load(file)
    else:
        os.makedirs(os.path.dirname(labeled_data_path), exist_ok=True)
        datasets: Dict[str, str] = {
            "train": os.path.join(thesis_path, "data", "wnut",
                                  "wnut_train.json"),
            "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
            "test": os.path.join(thesis_path, "data", "wnut",
                                 "wnut_test.json"),
        }

        def get_gazetteer_content(gazetteer_comb):
            from haystack.document_stores import ElasticsearchDocumentStore
            from search.utils import EMBEDDING_DIM
            name = "_".join(gazetteer_comb)
            if name != "lownergaz_sent":
                sent_name = name + "_sent"
                lownergaz_name = name + "_lownergaz"
            else:
                sent_name = "sent"
                lownergaz_name = "lownergaz"
            gazetteer_content = []
            for index in [sent_name, lownergaz_name]:
                doc_store = ElasticsearchDocumentStore(
                    index=index,
                    embedding_dim=EMBEDDING_DIM,
                    similarity="cosine")
                gazetteer_content.extend([{
                    "content": str(doc.content),
                    "meta": doc.meta
                } for doc in doc_store.get_all_documents()])

            return gazetteer_content

        labeled_data = defaultdict(list)

        unique_gazetteer_contents = set(
            [cont for gaz_cont in gazetteer_comb for cont in gaz_cont])

        for gaz_comb in unique_gazetteer_contents:
            labeled_data[gaz_comb] = get_gazetteer_content(gaz_comb)

        with open(labeled_data_path, "wb") as file:
            pickle.dump((labeled_data, datasets), file)

    return labeled_data, datasets, combination_to_content


def get_labeled_data_entity_coverage():
    eecr_labeled_data_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "03_emerging_labeled_data_eecr_metrics.parquet")
    if os.path.exists(eecr_labeled_data_data_path):
        eecr_df = pd.read_parquet(eecr_labeled_data_data_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(eecr_labeled_data_data_path),
                    exist_ok=True)
        labeled_data, datasets, combination_to_content = get_labeled_data()
        with mp.Pool(mp.cpu_count() - 5) as pool:
            with tqdm(total=len(labeled_data),
                      desc="Compute EECR labeled data") as pbar:
                results = pool.starmap_async(
                    get_ecr_labeled_data_df,
                    [(comb, data, datasets, combination_to_content)
                     for comb, data in labeled_data.items()],
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
        "03_emerging_labeled_data_eecr_per_sample_metrics.parquet")
    if os.path.exists(eecr_labeled_data_data_path):
        eecr_df = pd.read_parquet(eecr_labeled_data_data_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(eecr_labeled_data_data_path),
                    exist_ok=True)
        labeled_data, datasets, combination_to_content = get_labeled_data()
        with mp.Pool(mp.cpu_count() - 5) as pool:
            with tqdm(total=len(labeled_data),
                      desc="Compute EECR labeled data per sample") as pbar:
                results = pool.starmap_async(
                    get_ecr_per_sample_labeled_data_df,
                    [(comb, data, datasets, combination_to_content)
                     for comb, data in labeled_data.items()],
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


def get_ecr_labeled_data_df(comb, labeled_data: List[dict],
                            datasets: Dict[str, str],
                            combination_to_content: Dict[str,
                                                         List[Tuple[int,
                                                                    int]]]):
    eecr_metrics = []
    for content in combination_to_content[comb]:
        gazetteer_content = "None"
        if content[0] == 0:
            gazetteer_content = "lownergaz_sent"
        if content[0] == 1:
            gazetteer_content = "wnut_train"
        if content[0] == 2:
            gazetteer_content = "lownergaz_sent+wnut_train"
        timestep = content[1]
        for part, dataset_path in datasets.items():
            with open(dataset_path, "r") as file:
                dataset = json.load(file)
            ratio, c, eecr, _ = entity_coverage_ratio(labeled_data, dataset)
            ecr_classes = calc_ecr_classes(ratio, c)
            eecr_metrics.append({
                "gazetteer_content": gazetteer_content,
                "timestep": timestep,
                "dataset": "wnut_" + part,
                "eecr": eecr,
                **{key: len(value)
                   for key, value in ecr_classes.items()}
            })
    eecr_df = pd.DataFrame.from_records(eecr_metrics)
    return eecr_df


def get_ecr_per_sample_labeled_data_df(
        comb, labeled_data: List[dict], datasets: Dict[str, str],
        combination_to_content: Dict[str, List[Tuple[int, int]]]):

    eecr_metrics = []
    labeled_data_counted = count_entities(labeled_data)
    for content in combination_to_content[comb]:
        gazetteer_content = "None"
        if content[0] == 0:
            gazetteer_content = "lownergaz_sent"
        if content[0] == 1:
            gazetteer_content = "wnut_train"
        if content[0] == 2:
            gazetteer_content = "lownergaz_sent+wnut_train"
        timestep = content[1]
        for part, dataset_path in datasets.items():
            with open(dataset_path, "r") as file:
                dataset = json.load(file)
            for sample in dataset:
                sample_counted = count_entities([sample])
                ratio, c, eecr, _ = entity_coverage_ratio_precounted(
                    labeled_data_counted, sample_counted)
                ecr_classes = calc_ecr_classes(ratio, c)
                eecr_metrics.append({
                    "gazetteer_content": gazetteer_content,
                    "timestep": timestep,
                    "dataset": "wnut_" + part,
                    "doc_id": sample["doc_id"],
                    "targets": len(sample["entities"]),
                    "eecr": eecr,
                    **{key: len(value)
                       for key, value in ecr_classes.items()}
                })
    eecr_df = pd.DataFrame.from_records(eecr_metrics)
    return eecr_df


def get_search_results_data():
    search_results: List[str] = list(
        glob(os.path.join(thesis_path, "experiments",
                          "03_adaptation_emerging_entities", "data", r"**",
                          "01_search_results", "**", r"*.pkl"),
             recursive=True))
    datasets: Dict[str, str] = {
        "train": os.path.join(thesis_path, "data", "wnut", "wnut_train.json"),
        "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
        "test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
    }

    return search_results, datasets


def get_ecr_per_search_sample_df(search_result_path, datasets):
    fp = search_result_path.split(os.path.sep)
    eval_point = fp[-2].split("_")
    gazetteer_content = "None"
    if eval_point[0] == 0:
        gazetteer_content = "lownergaz_sent"
    if eval_point[0] == 1:
        gazetteer_content = "wnut_train"
    if eval_point[0] == 2:
        gazetteer_content = "lownergaz_sent+wnut_train"
    timestep = int(eval_point[1])
    name = os.path.splitext(fp[-1])[0].split("_")
    search = {}
    with open(search_result_path, "rb") as file:
        for idx, items in pickle.load(file).items():
            items: List[Document]
            search[idx] = [item.to_dict() for item in items]
    with open(datasets[name[-1]], "r") as file:
        dataset = json.load(file)
    eecr_metrics = []

    for idx, sample in enumerate(dataset):
        ratio, c, eecr, _ = entity_coverage_ratio(search[idx], [sample])
        ecr_classes = calc_ecr_classes(ratio, c)
        eecr_metrics.append({
            "gazetteer_content": gazetteer_content,
            "timestep": timestep,
            "dataset": "wnut_" + name[-1],
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
        "03_emerging_search_results_eecr_metrics.parquet")
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
                    [(search_result_path, datasets)
                     for search_result_path in search_results],
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
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-mpnet-base-v2" if device == "cuda" else "sentence-transformers/all-MiniLM-L6-v2"
    global model
    model = SearchSampleSimilarity(model_name).to(device)
    global collator
    collator = SearchSampleSimilarityCollator(model_name)


def get_ccr_per_sample_df(search_result_path, datasets):
    fp = search_result_path.split(os.path.sep)
    eval_point = fp[-2].split("_")
    gazetteer_content = "None"
    if eval_point[0] == 0:
        gazetteer_content = "lownergaz_sent"
    if eval_point[0] == 1:
        gazetteer_content = "wnut_train"
    if eval_point[0] == 2:
        gazetteer_content = "lownergaz_sent+wnut_train"
    timestep = int(eval_point[1])
    name = os.path.splitext(fp[-1])[0].split("_")
    search = {}
    with open(search_result_path, "rb") as file:
        for idx, items in pickle.load(file).items():
            items: List[Document]
            search[idx] = [item.to_dict() for item in items]
    with open(datasets[name[-1]], "r") as file:
        dataset = json.load(file)
    ccr_metrics = []
    for ccr in get_search_sample_similarity_for_model(model, collator, dataset,
                                                      search):
        ccr_metrics.append({
            "gazetteer_content": gazetteer_content,
            "timestep": timestep,
            "dataset": "wnut_" + name[-1],
            **ccr
        })
    ccr_df = pd.DataFrame.from_records(ccr_metrics)
    return ccr_df


def get_search_results_data_ccr_metrics():
    metrics_file_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "03_emerging_search_results_ccr_metrics.parquet")
    if os.path.exists(metrics_file_path):
        ccr_metrics_df = pd.read_parquet(metrics_file_path)
    else:
        pqwriter = None
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        search_results, datasets = get_search_results_data()
        with mp.get_context("spawn").Pool(1, get_ccr_init) as pool:
            global pbar
            with tqdm(total=len(search_results),
                      desc="CCR search results") as pbar:
                results = pool.starmap_async(
                    get_ccr_per_sample_df,
                    [(search_result_path, datasets)
                     for search_result_path in search_results],
                    callback=lambda _: pbar.update(1))
                for ccr_df in tqdm(results.get(),
                                   total=len(search_results),
                                   desc="Concat results to dataframe"):
                    table = pa.Table.from_pandas(ccr_df)
                    if pqwriter is None:
                        pqwriter = pq.ParquetWriter(metrics_file_path,
                                                    table.schema)
                    pqwriter.write_table(table)
                    pbar.update(1)
        if pqwriter is not None:
            pqwriter.close()

        ccr_metrics_df = pd.read_parquet(metrics_file_path)
    return ccr_metrics_df


def aggregate_performance_metrics(metrics_df: pd.DataFrame):
    agg_df = metrics_df.pivot_table(index=[
        "seed", "checkpoint", "has_gazetteer", "finetuning", "pretrained",
        "gazetteer_content", "timestep", "model", "dataset"
    ],
                                    values=["tp", "fp", "fn"],
                                    aggfunc="sum")
    agg_df["precision"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fp"])
    agg_df["recall"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fn"])
    agg_df["f1"] = 2 * agg_df["precision"] * agg_df["recall"] / (
        agg_df["precision"] + agg_df["recall"])

    return agg_df