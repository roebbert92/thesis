import copy
import pickle
import sys
import os
import json

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import multiprocessing as mp
from typing import Dict, List, Optional
from lightning.fabric.utilities.seed import seed_everything
from haystack import Pipeline
from haystack.nodes import JoinDocuments
from models.asp_t5 import get_tokenizer
from data_preprocessing.tokenize import query_database, tokenize_search_results_json
from search.lownergaz.setup import add_lownergaz_search_components
from search.sent.setup import add_sent_search_components
from configs.asp_t5 import T5_ASP_LOWNERGAZ_SENT
from data_augmentation.augments import make_erroneous_dataset, make_erroneous_gazetteer
from data_augmentation.sampling import per_type_uniform_sampling
from hyperparameter_tuning.utils import get_search_results_for_file
from search.utils import get_gazetteers_from_documents
from data_preparation.utils import remove_exact_matches


def get_tokenized_filepath(config, files, file_path, search_results, data_path,
                           output_name: Optional[str]):
    tokenizer = get_tokenizer(config)
    use_labels = config["use_labels"] if "use_labels" in config else None
    use_mentions = config["use_mentions"] if "use_mentions" in config else None
    sent_use_labels = config[
        "sent_use_labels"] if "sent_use_labels" in config else None
    sent_use_mentions = config[
        "sent_use_mentions"] if "sent_use_mentions" in config else None
    gaz_use_labels = config[
        "gaz_use_labels"] if "gaz_use_labels" in config else None
    gaz_use_mentions = config[
        "gaz_use_mentions"] if "gaz_use_mentions" in config else None

    return tokenize_search_results_json(tokenizer,
                                        file_path,
                                        files["types"],
                                        search_results,
                                        data_path,
                                        output_name=output_name,
                                        use_labels=use_labels,
                                        use_mentions=use_mentions,
                                        sent_use_labels=sent_use_labels,
                                        sent_use_mentions=sent_use_mentions,
                                        gaz_use_labels=gaz_use_labels,
                                        gaz_use_mentions=gaz_use_mentions,
                                        prepend_search_results=False)


def setup_database(sent_search_algorithm: str,
                   sent_search_topk: int,
                   gaz_search_algorithm: str,
                   gaz_search_topk: int,
                   join_method: str,
                   join_topk: int,
                   reset=False,
                   name: str = "lownergaz",
                   sents: List[dict] = [],
                   gazs: List[dict] = []):
    search = Pipeline()
    join_documents_input = []
    # sentences
    sent_name = "sent" if name == "lownergaz" else name + "_sent"
    add_sent_search_components(search, sent_search_algorithm, sent_search_topk,
                               join_documents_input, reset, sent_name, sents)

    # lowner gazetteers
    lownergaz_name = name if name == "lownergaz" else name + "_lownergaz"
    add_lownergaz_search_components(search, gaz_search_algorithm,
                                    gaz_search_topk, join_documents_input,
                                    reset, lownergaz_name, gazs)

    # join documents

    join_documents = JoinDocuments(join_mode=join_method, top_k_join=join_topk)
    search.add_node(join_documents, "DocumentJoin", join_documents_input)

    return search


# Datapoints
## 00_datasets
## 01_search_results
## 02_tokenized_datasets
## 03_checkpoints
## 04_metrics
def generate_experiment_data(seeds: List[int], gazetteer_sizes: List[int],
                             error_percent_ratios: List[int]):
    files = {
        "types":
        os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
        "train":
        os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
        "dev":
        os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
        "test":
        os.path.join(thesis_path, "data", "mlowner", "lowner_test.json"),
        "multiconer":
        os.path.join(thesis_path, "data", "multiconer",
                     "multiconer_test.json"),
        "lownergaz":
        os.path.join(thesis_path, "data", "mlowner", "lowner_gazetteer.json"),
    }

    cpu_count = int(mp.cpu_count() // 2)

    files["filtered_multiconer"] = os.path.join(
        os.path.dirname(files["multiconer"]), "filtered_multiconer.json")

    if not os.path.exists(files["filtered_multiconer"]):
        with open(files["train"], encoding="utf-8") as file:
            lowner_train = json.load(file)

        with open(files["dev"], encoding="utf-8") as file:
            lowner_dev = json.load(file)

        with open(files["test"], encoding="utf-8") as file:
            lowner_test = json.load(file)

        with open(files["multiconer"], encoding="utf-8") as file:
            multiconer = json.load(file)
        multiconer = remove_exact_matches(
            multiconer, lowner_train + lowner_dev + lowner_test)
        with open(files["filtered_multiconer"], "w", encoding="utf-8") as file:
            json.dump(multiconer, file)

    experiment_data_paths = {}
    config = T5_ASP_LOWNERGAZ_SENT
    config.update({
        "data_path":
        os.path.join(thesis_path, "experiments", "02_content", "data")
    })

    # 00_datasets
    experiment_data_paths["00_datasets"] = {}
    dataset_configs = [
        (config, files, seed, gazetteer_size, error_percent_ratio)
        for gazetteer_size in gazetteer_sizes
        for error_percent_ratio in error_percent_ratios for seed in seeds
    ]
    with mp.get_context('spawn').Pool(cpu_count // 2) as pool:
        for seed, gazetteer_size, error_percent_ratio, dataset_paths in pool.starmap(
                create_dataset, dataset_configs):
            experiment_data_paths["00_datasets"][
                f"{seed}_{gazetteer_size}_{error_percent_ratio}"] = dataset_paths

    # 01_search_results
    experiment_data_paths["01_search_results"] = {}
    search_results_configs = [
        (config, files, experiment_data_paths["00_datasets"]
         [f"{seed}_{gazetteer_size}_{error_percent_ratio}"], seed,
         gazetteer_size, error_percent_ratio)
        for gazetteer_size in gazetteer_sizes
        for error_percent_ratio in error_percent_ratios for seed in seeds
    ]
    with mp.get_context('spawn').Pool(cpu_count // 2) as pool:
        for seed, gazetteer_size, error_percent_ratio, search_result_paths in pool.starmap(
                get_search_results, search_results_configs):
            experiment_data_paths["01_search_results"][
                f"{seed}_{gazetteer_size}_{error_percent_ratio}"] = search_result_paths

    # setup database with full gazetteer
    search_base_path = os.path.join(config["data_path"], "01_search_results")
    os.makedirs(search_base_path, exist_ok=True)
    full_search = setup_database(config["sent_search_algorithm"],
                                 config["sent_search_topk"],
                                 config["gaz_search_algorithm"],
                                 config["gaz_search_topk"],
                                 config["search_join_method"],
                                 config["search_topk"])

    # get search results for clean lowner train, dev, test
    for part in ["train", "dev", "test"]:
        file_path_part = os.path.join(search_base_path,
                                      f"search_results_{part}.pkl")
        if not os.path.exists(file_path_part):
            search_results = get_search_results_for_file(
                full_search, files[part])
            with open(file_path_part, "wb") as file:
                pickle.dump(search_results, file)
        for gazetteer_size in gazetteer_sizes:
            for error_percent_ratio in error_percent_ratios:
                for seed in seeds:
                    experiment_data_paths["01_search_results"][
                        f"{seed}_{gazetteer_size}_{error_percent_ratio}"][
                            f"full_search_result_{part}"] = file_path_part

    # 02_tokenized_dataset
    tokenizing_configs = [(config, files, experiment_data_paths["00_datasets"]
                           [f"{seed}_{gazetteer_size}_{error_percent_ratio}"],
                           experiment_data_paths["01_search_results"]
                           [f"{seed}_{gazetteer_size}_{error_percent_ratio}"],
                           seed, gazetteer_size, error_percent_ratio)
                          for gazetteer_size in gazetteer_sizes
                          for error_percent_ratio in error_percent_ratios
                          for seed in seeds]
    experiment_data_paths["02_tokenized_dataset"] = {}
    with mp.Pool(cpu_count) as pool:
        for seed, gazetteer_size, error_percent_ratio, tokenized_files in pool.starmap(
                prep_tokenized_dataset, tokenizing_configs):
            experiment_data_paths["02_tokenized_dataset"][
                f"{seed}_{gazetteer_size}_{error_percent_ratio}"] = tokenized_files

    return experiment_data_paths


def prep_tokenized_dataset(config, files, dataset_paths, search_results_paths,
                           seed: int, gazetteer_size: int,
                           error_percent_ratio: int):
    tokenized_data_path = os.path.join(config["data_path"],
                                       f"seed_{str(seed)}",
                                       "02_tokenized_dataset",
                                       f"size_{gazetteer_size}",
                                       f"error_ratio_{error_percent_ratio}")
    os.makedirs(tokenized_data_path, exist_ok=True)
    seed_everything(seed)
    tokenized_files = {}

    # prep error lowner train + dev, clean lowner test
    for part in ["train", "dev"]:
        with open(search_results_paths[f"error_search_results_error_{part}"],
                  "rb") as file:
            error_search_results_error_part = pickle.load(file)
        tokenized_files[f"error_search_error_{part}"] = get_tokenized_filepath(
            config, files, dataset_paths[f"error_lowner_{part}"],
            error_search_results_error_part, tokenized_data_path,
            f"error_search_error_{part}")
    with open(search_results_paths["error_search_test"], "rb") as file:
        error_search_results_test = pickle.load(file)
    tokenized_files["error_search_test"] = get_tokenized_filepath(
        config, files, files["test"], error_search_results_test,
        tokenized_data_path, "error_search_test")

    # prep sampled clean lowner train, dev, test
    for part in ["train", "dev", "test"]:
        with open(search_results_paths[f"sampled_search_results_{part}"],
                  "rb") as file:
            error_search_results_error_part = pickle.load(file)
        tokenized_files[f"sampled_search_{part}"] = get_tokenized_filepath(
            config, files, files[part], error_search_results_error_part,
            tokenized_data_path, f"sampled_search_{part}")

    # prep full clean lowner train, dev, test
    for part in ["train", "dev", "test"]:
        with open(search_results_paths[f"full_search_result_{part}"],
                  "rb") as file:
            error_search_results_error_part = pickle.load(file)
        tokenized_files[f"full_search_{part}"] = get_tokenized_filepath(
            config, files, files[part], error_search_results_error_part,
            tokenized_data_path, f"full_search_{part}")

    return seed, gazetteer_size, error_percent_ratio, tokenized_files


def get_search_results(config, files, dataset_paths, seed: int,
                       gazetteer_size: int, error_percent_ratio: int):
    search_base_path = os.path.join(config["data_path"], f"seed_{str(seed)}",
                                    "01_search_results",
                                    f"size_{gazetteer_size}",
                                    f"error_ratio_{error_percent_ratio}")
    os.makedirs(search_base_path, exist_ok=True)
    seed_everything(seed)
    search_results_paths = {}
    with open(dataset_paths["error_sampled_multiconer"]) as file:
        error_multiconer = json.load(file)
    with open(dataset_paths["error_sampled_lownergaz"]) as file:
        error_lownergaz = json.load(file)
    # setup database for errorneous multiconer + lownergaz
    error_sampled_search = setup_database(
        config["sent_search_algorithm"],
        config["sent_search_topk"],
        config["gaz_search_algorithm"],
        config["gaz_search_topk"],
        config["search_join_method"],
        config["search_topk"],
        reset=True,
        name=f"{seed}_{gazetteer_size}_{error_percent_ratio}_error_sampled",
        sents=error_multiconer,
        gazs=error_lownergaz)
    # get search results for erroneous lowner train + dev, clean lowner test
    for part in ["train", "dev"]:
        error_search_results_error_part = get_search_results_for_file(
            error_sampled_search, dataset_paths[f"error_lowner_{part}"])
        search_results_paths[
            f"error_search_results_error_{part}"] = os.path.join(
                search_base_path, f"error_search_results_error_{part}.pkl")
        with open(search_results_paths[f"error_search_results_error_{part}"],
                  "wb") as file:
            pickle.dump(error_search_results_error_part, file)
    error_search_results_test = get_search_results_for_file(
        error_sampled_search, files["test"])
    search_results_paths["error_search_results_test"] = os.path.join(
        search_base_path, "error_search_results_test.pkl")
    with open(search_results_paths["error_search_results_test"], "wb") as file:
        pickle.dump(error_search_results_test, file)

    # setup database for sampled multiconer + lownergaz
    with open(dataset_paths["clean_sampled_multiconer"]) as file:
        sampled_multiconer = json.load(file)
    with open(dataset_paths["clean_sampled_lownergaz"]) as file:
        sampled_lownergaz = json.load(file)
    sampled_search = setup_database(
        config["sent_search_algorithm"],
        config["sent_search_topk"],
        config["gaz_search_algorithm"],
        config["gaz_search_topk"],
        config["search_join_method"],
        config["search_topk"],
        reset=True,
        name=f"{seed}_{gazetteer_size}_{error_percent_ratio}_clean_sampled",
        sents=sampled_multiconer,
        gazs=sampled_lownergaz)
    # get search results for clean lowner train, dev, test
    for part in ["train", "dev", "test"]:
        sampled_search_results_part = get_search_results_for_file(
            sampled_search, files[part])
        search_results_paths[f"sampled_search_results_{part}"] = os.path.join(
            search_base_path, f"sampled_search_results_{part}.pkl")
        with open(search_results_paths[f"sampled_search_results_{part}"],
                  "wb") as file:
            pickle.dump(sampled_search_results_part, file)

    return seed, gazetteer_size, error_percent_ratio, search_results_paths


def create_dataset(config, files, seed: int, gazetteer_size: int,
                   error_percent_ratio: int):
    dataset_base_path = os.path.join(config["data_path"], f"seed_{seed}",
                                     "00_datasets", f"size_{gazetteer_size}",
                                     f"error_ratio_{error_percent_ratio}")
    os.makedirs(dataset_base_path, exist_ok=True)

    seed_everything(seed)
    dataset_paths = {}
    error_ratio = error_percent_ratio / 100

    with open(files["filtered_multiconer"], encoding="utf-8") as file:
        multiconer = json.load(file)
    with open(files["types"]) as file:
        types = json.load(file)["entities"]

    # sample from multiconer
    print("started sampling multiconer")
    sampled_multiconer, _ = per_type_uniform_sampling(multiconer, list(types),
                                                      gazetteer_size)
    dataset_paths["clean_sampled_multiconer"] = os.path.join(
        dataset_base_path, "clean_sampled_multiconer.json")
    with open(dataset_paths["clean_sampled_multiconer"], "w",
              encoding="utf-8") as file:
        json.dump(sampled_multiconer, file)
    print("finished sampling multiconer")
    # sample from lownergaz
    print("started sampling lownergaz")
    lowner_gaz_search = Pipeline()
    add_lownergaz_search_components(lowner_gaz_search,
                                    config["gaz_search_algorithm"], 5)
    sampled_multiconer_gaz = [{
        "extended": str(doc.content).split(" ")
    } for doc in get_gazetteers_from_documents(sampled_multiconer)]
    sampled_lownergaz = []
    sampled_lownergaz_ids = set()
    for _, results in query_database(sampled_multiconer_gaz,
                                     lowner_gaz_search):
        for result in results:
            if result.id not in sampled_lownergaz_ids:
                sampled_lownergaz_ids.add(result.id)
                tokens = [
                    token for token in str(result.content).split(" ") if token
                ]
                sampled_lownergaz.append({
                    "tokens":
                    tokens,
                    "entities": [{
                        "start": 0,
                        "end": len(tokens),
                        "type": result.meta["type"],
                    }],
                    "doc_id":
                    result.id
                })
    dataset_paths["clean_sampled_lownergaz"] = os.path.join(
        dataset_base_path, "clean_sampled_lownergaz.json")
    with open(dataset_paths["clean_sampled_lownergaz"], "w",
              encoding="utf-8") as file:
        json.dump(sampled_lownergaz, file)
    print("finished sampling lownergaz")

    # create erroneous lowner train, dev + gazetteer split
    print("creating erroneous lowner train, dev + gazetteer split")
    for part in ["train", "dev"]:
        with open(files[part]) as file:
            lowner_part = json.load(file)
        error_lowner_part = make_erroneous_dataset(lowner_part, list(types),
                                                   error_ratio)
        dataset_paths[f"error_lowner_{part}"] = os.path.join(
            dataset_base_path, f"error_lowner_{part}.json")
        with open(dataset_paths[f"error_lowner_{part}"], "w",
                  encoding="utf-8") as file:
            json.dump(error_lowner_part, file)

    error_multiconer = make_erroneous_dataset(sampled_multiconer, list(types),
                                              error_ratio)
    dataset_paths["error_sampled_multiconer"] = os.path.join(
        dataset_base_path, "error_sampled_multiconer.json")
    with open(dataset_paths["error_sampled_multiconer"], "w",
              encoding="utf-8") as file:
        json.dump(error_multiconer, file)
    error_lownergaz = make_erroneous_gazetteer(sampled_lownergaz, list(types),
                                               error_ratio)
    dataset_paths["error_sampled_lownergaz"] = os.path.join(
        dataset_base_path, "error_sampled_lownergaz.json")
    with open(dataset_paths["error_sampled_lownergaz"], "w",
              encoding="utf-8") as file:
        json.dump(error_lownergaz, file)

    print("finished erroneous lowner train, dev + gazetteer split")

    return seed, gazetteer_size, error_percent_ratio, dataset_paths


if __name__ == "__main__":
    seeds = [1, 2, 3]
    gazetteer_sizes = [2000, 4000, 6000, 8000]
    error_percent_ratios = [0, 5, 10, 15]
    experiment_data = generate_experiment_data(seeds, gazetteer_sizes,
                                               error_percent_ratios)
    with open("experiment_data_paths.json", "w") as file:
        json.dump(experiment_data, file)
    print("done")