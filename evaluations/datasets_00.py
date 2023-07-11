import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from collections import defaultdict
import json
import pickle
from typing import Dict, List
import pandas as pd
from glob import glob
from haystack import Document
from tqdm import tqdm
from data_metrics.entity_coverage_ratio import entity_coverage_ratio, count_entities, entity_coverage_ratio_precounted
from data_metrics.search_sample_similarity import get_search_sample_similarity
from collections import Counter
from statistics import mean, stdev


def calc_ecr_classes(ratio: dict, c: dict):
    return {
        "ρ=1": [key for key, value in ratio.items() if value == 1.0],
        "ρ ∈ (0.5,1)":
        [key for key, value in ratio.items() if 0.5 < value < 1.0],
        "ρ ∈ (0,0.5]":
        [key for key, value in ratio.items() if 0.0 < value <= 0.5],
        "ρ=0∧C≠0":
        [key for key, value in ratio.items() if value == 0.0 and c[key] != 0],
        "ρ=0∧C=0":
        [key for key, value in ratio.items() if value == 0.0 and c[key] == 0]
    }


def get_labeled_data():
    labeled_data_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "00_datasets_labeled_data.pkl.tar.gz")
    if os.path.exists(labeled_data_path):
        with open(labeled_data_path, "rb") as file:
            labeled_data, models_to_labeled_data, dataset = pickle.load(file)
    else:
        os.makedirs(os.path.dirname(labeled_data_path), exist_ok=True)
        labeled_data = defaultdict(list)
        for key in ["gaz", "lownergaz", "sent"]:
            if key in ["gaz", "sent"]:
                export_path = os.path.join(thesis_path, "data", "multiconer",
                                           f"multiconer_{key}.json")
            else:
                export_path = os.path.join(thesis_path, "data", "mlowner",
                                           "lowner_gazetteer_db.json")
            with open(export_path, "r") as file:
                for item in json.load(file):
                    labeled_data[key].append(item)

        models_to_labeled_data = {
            't5_asp_lownergaz': ["lownergaz"],
            't5_asp_gaz': ["gaz"],
            't5_asp_lownergaz_sent': ["lownergaz", "sent"],
            't5_asp_gaz_sent': ["gaz", "sent"],
            't5_asp_sent': ["sent"]
        }

        dataset = defaultdict(list)
        # lowner
        for key in ["lowner_train", "lowner_dev", "lowner_test"]:
            with open(os.path.join(thesis_path, "data", "mlowner",
                                   f"{key}.json"),
                      "r",
                      encoding="utf-8") as file:
                dataset[key].extend(json.load(file))

        # wnut
        for key in ["wnut_train", "wnut_dev", "wnut_test"]:
            with open(os.path.join(thesis_path, "data", "wnut", f"{key}.json"),
                      "r",
                      encoding="utf-8") as file:
                dataset[key].extend(json.load(file))
        with open(labeled_data_path, "wb") as file:
            pickle.dump((labeled_data, models_to_labeled_data, dataset), file)

    return labeled_data, models_to_labeled_data, dataset


def get_labeled_data_entity_coverage():
    eecr_labeled_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "00_datasets_labeled_data_eecr_metrics.pkl.tar.gz")
    if os.path.exists(eecr_labeled_data_path):
        eecr_df = pd.read_pickle(eecr_labeled_data_path)
    else:
        os.makedirs(os.path.dirname(eecr_labeled_data_path), exist_ok=True)
        labeled_data, models_to_labeled_data, dataset = get_labeled_data()
        eecr_metrics = []
        with tqdm(total=len(models_to_labeled_data) * len(dataset),
                  desc="EECR labeled data") as pbar:
            for model, data_parts in models_to_labeled_data.items():
                data = [
                    item for part in data_parts for item in labeled_data[part]
                ]
                for dataset_name, d in dataset.items():
                    ratio, c, eecr, _ = entity_coverage_ratio(data, d)
                    ecr_classes = calc_ecr_classes(ratio, c)
                    eecr_metrics.append({
                        "model": model,
                        "dataset": dataset_name,
                        "eecr": eecr,
                        **{
                            key: len(value)
                            for key, value in ecr_classes.items()
                        }
                    })
                    pbar.update(1)

        eecr_df = pd.DataFrame.from_records(eecr_metrics)
        eecr_df.to_pickle(eecr_labeled_data_path)
    return eecr_df


def get_labeled_data_entity_coverage_per_sample():
    eecr_labeled_data_per_sample_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "00_datasets_labeled_data_eecr_per_sample_metrics.pkl.tar.gz")
    if os.path.exists(eecr_labeled_data_per_sample_path):
        eecr_df = pd.read_pickle(eecr_labeled_data_per_sample_path)
    else:
        os.makedirs(os.path.dirname(eecr_labeled_data_per_sample_path),
                    exist_ok=True)
        labeled_data, models_to_labeled_data, dataset = get_labeled_data()
        eecr_metrics = []
        with tqdm(total=len(models_to_labeled_data) * len(dataset),
                  desc="EECR labeled data per sample") as pbar:
            for model, data_parts in models_to_labeled_data.items():
                data = [
                    item for part in data_parts for item in labeled_data[part]
                ]
                data_counted = count_entities(data)
                for dataset_name, d in dataset.items():
                    for sample in d:
                        sample_counted = count_entities([sample])
                        ratio, c, eecr, _ = entity_coverage_ratio_precounted(
                            data_counted, sample_counted)
                        ecr_classes = calc_ecr_classes(ratio, c)
                        eecr_metrics.append({
                            "model": model,
                            "dataset": dataset_name,
                            "doc_id": sample["doc_id"],
                            "targets": len(sample["entities"]),
                            "eecr": eecr,
                            **{
                                key: len(value)
                                for key, value in ecr_classes.items()
                            }
                        })
                    pbar.update(1)

        eecr_df = pd.DataFrame.from_records(eecr_metrics)
        eecr_df.to_pickle(eecr_labeled_data_per_sample_path)
    return eecr_df


def aggregate_per_sample_eecr_metrics(metrics_df: pd.DataFrame):
    return metrics_df.pivot_table(values=["eecr"],
                                  index=["model", "dataset"],
                                  aggfunc=["mean",
                                           "std"]).swaplevel(0, 1,
                                                             1).reset_index()


def get_search_results_data():
    search_results_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "00_datasets_search_results.pkl.tar.gz")
    if os.path.exists(search_results_data_path):
        with open(search_results_data_path, "rb") as file:
            search_results, dataset = pickle.load(file)
    else:
        os.makedirs(os.path.dirname(search_results_data_path), exist_ok=True)
        search_results = {}
        for data_path in glob(os.path.join(thesis_path, "experiments",
                                           "01_performance", "data",
                                           "01_search_results", "**", "*.pkl"),
                              recursive=True):
            fp = data_path.split(os.path.sep)
            model_name = fp[-2]
            dataset_name = os.path.splitext(fp[-1])[0]
            if model_name not in search_results:
                search_results[model_name] = defaultdict(dict)
            with open(data_path, "rb") as file:
                for idx, items in pickle.load(file).items():
                    items: List[Document]
                    search_results[model_name][dataset_name][idx] = [
                        item.to_dict() for item in items
                    ]

        dataset = defaultdict(list)
        # lowner
        for key in ["lowner_train", "lowner_dev", "lowner_test"]:
            with open(os.path.join(thesis_path, "data", "mlowner",
                                   f"{key}.json"),
                      "r",
                      encoding="utf-8") as file:
                dataset[key].extend(json.load(file))
        # wnut
        for key in ["wnut_train", "wnut_dev", "wnut_test"]:
            with open(os.path.join(thesis_path, "data", "wnut", f"{key}.json"),
                      "r",
                      encoding="utf-8") as file:
                dataset[key].extend(json.load(file))
        with open(search_results_data_path, "wb") as file:
            pickle.dump((search_results, dataset), file)
    return search_results, dataset


def get_search_results_entity_coverage_per_sample():
    eecr_search_results_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "00_datasets_search_results_eecr_metrics.pkl.tar.gz")
    if os.path.exists(eecr_search_results_data_path):
        eecr_df = pd.read_pickle(eecr_search_results_data_path)
    else:
        os.makedirs(os.path.dirname(eecr_search_results_data_path),
                    exist_ok=True)
        search_results, dataset = get_search_results_data()
        eecr_metrics = []
        with tqdm(total=len(search_results) * len(dataset),
                  desc="EECR search results") as pbar:
            for model, search_result in search_results.items():
                for dataset_name, search in search_result.items():
                    for idx, sample in enumerate(dataset[dataset_name]):
                        ratio, c, eecr, _ = entity_coverage_ratio(
                            search[idx], [sample])
                        ecr_classes = calc_ecr_classes(ratio, c)
                        eecr_metrics.append({
                            "model": model,
                            "dataset": dataset_name,
                            "doc_id": sample["doc_id"],
                            "targets": len(sample["entities"]),
                            "eecr": eecr,
                            **{
                                key: len(value)
                                for key, value in ecr_classes.items()
                            }
                        })
                    pbar.update(1)

        eecr_df = pd.DataFrame.from_records(eecr_metrics)
        eecr_df.to_pickle(eecr_search_results_data_path)
    return eecr_df


def get_entity_coverages(datasets: List[str]):
    # combine to one table
    labeled_data_eecr_df = get_labeled_data_entity_coverage()
    labeled_data_eecr_sample_df = get_labeled_data_entity_coverage_per_sample()
    agg_labeled_data_eecr_sample_df = aggregate_per_sample_eecr_metrics(
        labeled_data_eecr_sample_df)
    search_results_data_eecr = get_search_results_entity_coverage_per_sample()
    agg_search_results_data_eecr = aggregate_per_sample_eecr_metrics(
        search_results_data_eecr)

    eecr_table = labeled_data_eecr_df[["model", "dataset", "eecr"]].set_index([
        "model", "dataset"
    ]).join(agg_labeled_data_eecr_sample_df.set_index(["model", "dataset"]),
            on=["model", "dataset"],
            lsuffix="_labeled_data").join(
                agg_search_results_data_eecr.set_index(["model", "dataset"]),
                on=["model", "dataset"],
                lsuffix="_labeled_data_per_sample",
                rsuffix="_search_results").reset_index()
    return eecr_table[eecr_table["dataset"].isin(datasets)][[
        "model", "eecr_labeled_data", "eecr_labeled_data_per_sample",
        "eecr_search_results"
    ]]


def aggregate_per_sample_ecr_classes(metrics_df: pd.DataFrame):
    ECR_CLASSES_ORDER = [
        "ρ=1", "ρ ∈ (0.5,1)", "ρ ∈ (0,0.5]", "ρ=0∧C≠0", "ρ=0∧C=0"
    ]
    return metrics_df.pivot_table(
        values=ECR_CLASSES_ORDER, index=["model", "dataset"],
        aggfunc="sum")[ECR_CLASSES_ORDER].reset_index()


def get_search_results_data_ccr_metrics():
    metrics_file_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "00_datasets_search_results_ccr_metrics.pkl.tar.gz")
    if os.path.exists(metrics_file_path):
        ccr_metrics_df = pd.read_pickle(metrics_file_path)
    else:
        search_results, dataset = get_search_results_data()
        ccr_metrics = []
        # ccr_columns = [
        #     "max",
        #     "φ ∈ (0.5,1]",
        #     "φ ∈ (0,0.5]",
        #     "φ ∈ (-0.5,0]",
        #     "φ ∈ [-1,-0.5]",
        # ]
        with tqdm(total=len(search_results) * len(dataset),
                  desc="CCR search results") as pbar:
            for model, search_result in search_results.items():
                for dataset_name, search in search_result.items():
                    for ccr in get_search_sample_similarity(
                            dataset[dataset_name], search):
                        ccr_metrics.append({
                            "model": model,
                            "dataset": dataset_name,
                            **ccr
                        })
                    pbar.update(1)

        ccr_metrics_df = pd.DataFrame.from_records(ccr_metrics)
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        ccr_metrics_df.to_pickle(metrics_file_path)
    return ccr_metrics_df


def get_dataset_stats_df(files: Dict[str, Dict[str, str]]):
    dataset_stats = []
    for dataset, split in files.items():
        for part, filepath in split.items():
            with open(filepath) as file:
                samples = json.load(file)
            total_samples = len(samples)
            if "tokens" in samples[0]:
                sample_lens = [len(sample["tokens"]) for sample in samples]
                sample_entity_count = [
                    len(sample["entities"]) for sample in samples
                ]
                entities = [
                    ent["type"] for sample in samples
                    for ent in sample["entities"]
                ]
            else:
                sample_lens = [
                    len(sample["entity"].split(" ")) for sample in samples
                ]
                sample_entity_count = [1 for _ in samples]
                entities = [sample["type"] for sample in samples]
            entity_count = dict(Counter(entities))
            dataset_stats.append({
                "dataset": "_".join([dataset, part]),
                "total samples": total_samples,
                "sample len avg": mean(sample_lens),
                "sample len std": stdev(sample_lens),
                "total entities": sum(sample_entity_count),
                "entity count avg": mean(sample_entity_count),
                "entity count std": stdev(sample_entity_count),
                **entity_count
            })

    return pd.DataFrame.from_records(dataset_stats)


def get_gazetteer_stats_df(files: Dict[str, List[str]]):
    gazetteer_stats = []
    for gazetteer_name, filepaths in files.items():
        samples = []
        for filepath in filepaths:
            with open(filepath) as file:
                samples.extend(json.load(file))
        total_samples = len(samples)
        sample_lens = []
        sample_entity_count = []
        entities = []
        distinct_entities = []
        for sample in samples:
            if "entities" in sample:
                sample_lens.append(len(sample["tokens"]))
                sample_entity_count.append(len(sample["entities"]))
                entities.extend([ent["type"] for ent in sample["entities"]])
                distinct_entities.extend([
                    frozenset([
                        ent["type"],
                        " ".join(sample["tokens"][ent["start"]:ent["end"]])
                    ]) for ent in sample["entities"]
                ])
            elif "entity" in sample:
                sample_lens.append(len(sample["entity"].split(" ")))
                sample_entity_count.append(1)
                entities.append(sample["type"])
                distinct_entities.append(
                    frozenset([sample["type"], sample["entity"]]))
            elif "content" in sample:
                tokens = sample["content"].strip().split(" ")
                sample_lens.append(len(tokens))
                if "entities" in sample["meta"]:
                    sample_entity_count.append(len(sample["meta"]["entities"]))
                    entities.extend(
                        [ent["type"] for ent in sample["meta"]["entities"]])
                    distinct_entities.extend([
                        frozenset([
                            ent["type"],
                            " ".join(tokens[ent["start"]:ent["end"]])
                        ]) for ent in sample["meta"]["entities"]
                    ])
                elif "type" in sample["meta"]:
                    sample_entity_count.append(1)
                    entities.append(sample["meta"]["type"])
                    distinct_entities.append(
                        frozenset([sample["meta"]["type"], sample["content"]]))
        entity_count = dict(Counter(entities))
        distinct_entity_count = dict(Counter(distinct_entities))
        gazetteer_stats.append({
            "gazetteer": gazetteer_name,
            "total samples": total_samples,
            "sample len avg": mean(sample_lens),
            "sample len std": stdev(sample_lens),
            "total entities": sum(sample_entity_count),
            "distinct entities": len(distinct_entity_count),
            "entity count avg": mean(sample_entity_count),
            "entity count std": stdev(sample_entity_count),
            **entity_count
        })

    return pd.DataFrame.from_records(gazetteer_stats)