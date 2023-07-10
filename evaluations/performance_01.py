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
import json
import pickle
from haystack import Document
from collections import defaultdict
from data_metrics.entity_coverage_ratio import entity_coverage_ratio, entity_coverage_ratio_precounted, count_entities
from data_metrics.search_sample_similarity import get_search_sample_similarity
from tqdm import tqdm
from evaluations.utils import MODEL_ORDER, PLOT_MODEL_NAMES, PLOT_SEARCH_NAMES
import matplotlib.pyplot as plt

plt.rcParams.update({
    'mathtext.default': 'regular',
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 14
})


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
    metrics_file_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "01_performance_per_sample_metrics.pkl.tar.gz")
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


def aggregate_error_rate(metrics_df: pd.DataFrame, checkpoint: str,
                         dataset: str):
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset"],
        values=["tp", "fp", "fn"],
        aggfunc="sum")
    agg_df["total errors"] = agg_df["fp"] + agg_df["fn"]
    return agg_df


def aggregate_performance_metrics(metrics_df: pd.DataFrame):
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset"],
        values=["tp", "fp", "fn"],
        aggfunc="sum")
    agg_df["precision"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fp"])
    agg_df["recall"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fn"])
    agg_df["f1"] = 2 * agg_df["precision"] * agg_df["recall"] / (
        agg_df["precision"] + agg_df["recall"])

    return agg_df


def get_micro_f1_precision_recall(metrics_df: pd.DataFrame, checkpoint: str,
                                  dataset: str):
    agg_df = aggregate_performance_metrics(metrics_df)

    avg_df = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                values=["f1", "precision", "recall"],
                                aggfunc=("mean", "std")).reset_index()

    return avg_df[(avg_df["checkpoint"] == checkpoint)
                  & (avg_df["dataset"] == dataset)]


def get_precision(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    agg_df = aggregate_performance_metrics(metrics_df)

    avg_precision = agg_df.pivot_table(
        index=["model", "checkpoint", "dataset"],
        values=["precision"],
        aggfunc=("mean", "std")).reset_index()

    return avg_precision[(avg_precision["checkpoint"] == checkpoint)
                         & (avg_precision["dataset"] == dataset)]


def get_recall(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    agg_df = aggregate_performance_metrics(metrics_df)

    avg_recall = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                    values=["recall"],
                                    aggfunc=("mean", "std")).reset_index()

    return avg_recall[(avg_recall["checkpoint"] == checkpoint)
                      & (avg_recall["dataset"] == dataset)]


def get_micro_f1(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    agg_df = aggregate_performance_metrics(metrics_df)

    avg_f1 = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                values=["f1"],
                                aggfunc=("mean", "std")).reset_index()

    return avg_f1[(avg_f1["checkpoint"] == checkpoint)
                  & (avg_f1["dataset"] == dataset)]


def get_error_types(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    error_types = [
        "error_type1", "error_type2", "error_type3", "error_type4",
        "error_type5"
    ]
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset"],
        values=error_types,
        aggfunc="sum")
    agg_df["sum"] = agg_df[error_types].sum(axis=1)
    avg_errors = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                    values=["sum", *error_types],
                                    aggfunc=("mean", "std")).reset_index()

    return avg_errors[(avg_errors["checkpoint"] == checkpoint)
                      & (avg_errors["dataset"] == dataset)]


def get_f1_per_entity(metrics_df: pd.DataFrame, checkpoint: str, dataset: str):
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset", "entity_type"],
        values=["tp", "fp", "fn"],
        aggfunc="sum")
    agg_df["precision"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fp"])
    agg_df["recall"] = 100 * agg_df["tp"] / (agg_df["tp"] + agg_df["fn"])
    agg_df["f1"] = 2 * agg_df["precision"] * agg_df["recall"] / (
        agg_df["precision"] + agg_df["recall"])

    avg_f1 = agg_df.pivot_table(
        index=["model", "checkpoint", "dataset"],
        columns=["entity_type"],
        values=["f1"],
        aggfunc=("mean", "std")).swaplevel(axis=1, i=2, j=0).sort_index(
            axis=1, level=2, sort_remaining=True).reset_index()

    return avg_f1[(avg_f1["checkpoint"] == checkpoint)
                  & (avg_f1["dataset"] == dataset)]


def get_error_types_per_entity(metrics_df: pd.DataFrame, checkpoint: str,
                               dataset: str):
    error_types = [
        "error_type1", "error_type2", "error_type3", "error_type4",
        "error_type5"
    ]
    agg_df = metrics_df.pivot_table(
        index=["seed", "model", "checkpoint", "dataset", "entity_type"],
        values=error_types,
        aggfunc="sum")

    avg_errors = agg_df.pivot_table(index=["model", "checkpoint", "dataset"],
                                    columns=["entity_type"],
                                    values=error_types,
                                    aggfunc=("mean", "std")).swaplevel(
                                        axis=1, i=2, j=0).sort_index(
                                            axis=1,
                                            level=2,
                                            sort_remaining=True).reset_index()

    return avg_errors[(avg_errors["checkpoint"] == checkpoint)
                      & (avg_errors["dataset"] == dataset)]


def get_labeled_data():
    labeled_data_path = os.path.join(thesis_path, "evaluations", "metrics",
                                     "01_performance_labeled_data.pkl.tar.gz")
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
        for key in ["lowner_train", "lowner_dev", "lowner_test"]:
            with open(os.path.join(thesis_path, "data", "mlowner",
                                   f"{key}.json"),
                      "r",
                      encoding="utf-8") as file:
                dataset[key].extend(json.load(file))
        with open(labeled_data_path, "wb") as file:
            pickle.dump((labeled_data, models_to_labeled_data, dataset), file)

    return labeled_data, models_to_labeled_data, dataset


def get_search_results_data():
    search_results_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "01_performance_search_results_data.pkl.tar.gz")
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
            if not dataset_name.startswith("lowner"):
                continue
            if model_name not in search_results:
                search_results[model_name] = defaultdict(dict)
            with open(data_path, "rb") as file:
                for idx, items in pickle.load(file).items():
                    items: List[Document]
                    search_results[model_name][dataset_name][idx] = [
                        item.to_dict() for item in items
                    ]
        dataset = defaultdict(list)
        for key in ["lowner_train", "lowner_dev", "lowner_test"]:
            with open(os.path.join(thesis_path, "data", "mlowner",
                                   f"{key}.json"),
                      "r",
                      encoding="utf-8") as file:
                dataset[key].extend(json.load(file))
        with open(search_results_data_path, "wb") as file:
            pickle.dump((search_results, dataset), file)
    return search_results, dataset


def get_labeled_data_entity_coverage():
    eecr_labeled_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "01_performance_labeled_data_eecr_metrics.pkl.tar.gz")
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
        "01_performance_labeled_data_eecr_per_sample_metrics.pkl.tar.gz")
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


def get_search_results_entity_coverage_per_sample():
    eecr_search_results_data_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "01_performance_search_results_eecr_metrics.pkl.tar.gz")
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


def aggregate_per_sample_eecr_metrics(metrics_df: pd.DataFrame):
    return metrics_df.pivot_table(values=["eecr"],
                                  index=["model", "dataset"],
                                  aggfunc=["mean",
                                           "std"]).swaplevel(0, 1,
                                                             1).reset_index()


def aggregate_per_sample_ecr_classes(metrics_df: pd.DataFrame):
    ECR_CLASSES_ORDER = [
        "ρ=1", "ρ ∈ (0.5,1)", "ρ ∈ (0,0.5]", "ρ=0∧C≠0", "ρ=0∧C=0"
    ]
    return metrics_df.pivot_table(
        values=ECR_CLASSES_ORDER, index=["model", "dataset"],
        aggfunc="sum")[ECR_CLASSES_ORDER].reset_index()


def get_entity_coverages(dataset: str):
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
    return eecr_table[eecr_table["dataset"] == dataset][[
        "model", "eecr_labeled_data", "eecr_labeled_data_per_sample",
        "eecr_search_results"
    ]]


def get_ecr_plotable_table(ecr_df: pd.DataFrame, dataset: str):
    plt_ecr_df = ecr_df[ecr_df["dataset"] ==
                        dataset].loc[:,
                                     ~ecr_df.columns.isin(["eecr", "dataset"])]
    plt_ecr_df = plt_ecr_df.sort_values(
        "model", key=lambda x: x.apply(lambda y: MODEL_ORDER.get(y, 1000)))
    plt_ecr_df["Search"] = plt_ecr_df["model"].apply(
        lambda x: PLOT_SEARCH_NAMES[x])
    plt_ecr_df = plt_ecr_df.set_index("Search")
    return plt_ecr_df.loc[:, ~plt_ecr_df.columns.isin(["model"])].T


def get_search_results_data_ccr_metrics():
    metrics_file_path = os.path.join(
        thesis_path, "evaluations", "metrics",
        "01_performance_search_results_ccr_metrics.pkl.tar.gz")
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


def get_search_results_data_ccr_max():
    ccr_metrics_df = get_search_results_data_ccr_metrics()

    return ccr_metrics_df.pivot_table(values="max",
                                      index=["model", "dataset"],
                                      aggfunc=["mean", "std"]).reset_index()
