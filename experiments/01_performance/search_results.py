import sys
import os

import torch

thesis_path = '/home/loebbert/projects/thesis'
sys.path.append(thesis_path)

from configs.asp_t5 import T5_BASE, FLAN_T5_BASE
import json
from hyperparameter_tuning.t5_asp_fetahugaz_sent import setup_database as setup_database_fetahugaz_sent
from hyperparameter_tuning.t5_asp_gaz_sent import setup_database as setup_database_gaz_sent
from hyperparameter_tuning.t5_asp_fetahugaz import setup_database as setup_database_fetahugaz
from hyperparameter_tuning.t5_asp_gaz import setup_database as setup_database_gaz
from hyperparameter_tuning.t5_asp_sent import setup_database as setup_database_sent
from hyperparameter_tuning.training import get_search_results
from lightning.fabric.utilities.seed import seed_everything
import pickle
from data_metrics.entity_coverage_ratio import entity_coverage_ratio
from data_metrics.sample_similarity import get_search_sample_similarity
import pandas as pd
from tqdm import tqdm

t5_asp_fetahugaz_sent = T5_BASE.copy()
t5_asp_fetahugaz_sent.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "gaz_search_algorithm": "bm25",
    "gaz_search_topk": 6,
    "gaz_use_mentions": False,
    "num_epochs": 16,
    "plm_learning_rate": 0.00017496219281663535,
    "search_join_method": "reciprocal_rank_fusion",
    "search_topk": 8,
    "sent_search_algorithm": "ann",
    "sent_search_topk": 6,
    "sent_use_mentions": True,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_fetahugaz_sent"
})

t5_asp_fetahugaz = FLAN_T5_BASE.copy()
t5_asp_fetahugaz.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "num_epochs": 16,
    "plm_learning_rate": 0.00017496219281663535,
    "search_algorithm": "bm25",
    "search_topk": 8,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "use_mentions": False,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_fetahugaz"
})

t5_asp_gaz = FLAN_T5_BASE.copy()
t5_asp_gaz.update({
    "adam_weight_decay": 0.018862500000000015,
    "asp_dropout_rate": 0.43875,
    "asp_hidden_dim": 799,
    "num_epochs": 17,
    "plm_learning_rate": 0.00020887755102040807,
    "search_algorithm": "bm25",
    "search_topk": 6,
    "task_learning_rate": 0.003949473684210526,
    "train_search_dropout": 0.028260869565217374,
    "use_mentions": False,
    "warmup_ratio": 0.20864864864864865,
    "name": "t5_asp_gaz"
})

t5_asp_gaz_sent = T5_BASE.copy()
t5_asp_gaz_sent.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "gaz_search_algorithm": "bm25",
    "gaz_search_topk": 6,
    "gaz_use_mentions": False,
    "num_epochs": 24,
    "plm_learning_rate": 0.00017496219281663535,
    "search_join_method": "reciprocal_rank_fusion",
    "search_topk": 8,
    "sent_search_algorithm": "ann",
    "sent_search_topk": 6,
    "sent_use_mentions": True,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_gaz_sent"
})

t5_asp_sent = T5_BASE.copy()
t5_asp_sent.update({
    "adam_weight_decay": 0.49637507889057786,
    "asp_dropout_rate": 0.3,
    "asp_hidden_dim": 142,
    "num_epochs": 20,
    "plm_learning_rate": 5e-05,
    "search_algorithm": "ann",
    "search_topk": 8,
    "task_learning_rate": 0.0013480523331922776,
    "train_search_dropout": 0.21126587935893093,
    "use_mentions": True,
    "warmup_ratio": 0.184451637360714,
    "name": "t5_asp_sent"
})

with open("/home/loebbert/projects/thesis/data/mlowner/lowner_train.json",
          encoding="utf-8") as file:
    lowner_train = json.load(file)

with open("/home/loebbert/projects/thesis/data/mlowner/lowner_dev.json",
          encoding="utf-8") as file:
    lowner_dev = json.load(file)

with open("/home/loebbert/projects/thesis/data/mlowner/lowner_test.json",
          encoding="utf-8") as file:
    lowner_test = json.load(file)

seeds = [1, 2, 3]
search_result_eecr = []
search_result_ccr = []

for seed in seeds:
    # seed
    seed_everything(seed)

    for config in [
            t5_asp_fetahugaz_sent, t5_asp_fetahugaz, t5_asp_gaz,
            t5_asp_gaz_sent, t5_asp_sent
    ]:

        # setup database
        if config["name"] == "t5_asp_fetahugaz_sent":
            search = setup_database_fetahugaz_sent(
                config["sent_search_algorithm"],
                config["sent_search_topk"],
                config["gaz_search_algorithm"],
                config["gaz_search_topk"],
                config["search_join_method"],
                config["search_topk"],
            )
        if config["name"] == "t5_asp_gaz_sent":
            search = setup_database_gaz_sent(
                config["sent_search_algorithm"],
                config["sent_search_topk"],
                config["gaz_search_algorithm"],
                config["gaz_search_topk"],
                config["search_join_method"],
                config["search_topk"],
            )
        if config["name"] == "t5_asp_fetahugaz":
            search = setup_database_fetahugaz(config["search_algorithm"],
                                              config["search_topk"])
        if config["name"] == "t5_asp_gaz":
            search = setup_database_gaz(config["search_algorithm"],
                                        config["search_topk"])
        if config["name"] == "t5_asp_sent":
            search = setup_database_sent(config["search_algorithm"],
                                         config["search_topk"])

        # go through all datasets
        for dataset, dataset_name in [(lowner_train, "lowner_train"),
                                      (lowner_dev, "lowner_dev"),
                                      (lowner_test, "lowner_test")]:

            # save search results for augmentation
            file_name = os.path.join(
                "/home/loebbert/projects/thesis", "experiments",
                "01_performance", "search_results",
                f"{str(seed)}_{config['name']}_{dataset_name}.pkl")

            if not os.path.exists(file_name):
                search_results = get_search_results(search, dataset)
                with open(file_name, "wb") as file:
                    pickle.dump(search_results, file)
            else:
                with open(file_name, "rb") as file:
                    search_results = pickle.load(file)

            # calculate the expected entity coverage ratio for each sample
            for idx, item in tqdm(enumerate(dataset),
                                  total=len(dataset),
                                  desc="EECR"):
                results = [doc.to_dict() for doc in search_results[idx]]
                _, _, eecr = entity_coverage_ratio(results, [item])
                search_result_eecr.append({
                    "dataset": dataset_name,
                    "model": config["name"],
                    "doc_id": item["doc_id"],
                    "seed": seed,
                    "eecr": eecr
                })

            # calculate the context coverage ratio for each sample
            for result in tqdm(get_search_sample_similarity(
                    dataset, search_results),
                               total=len(dataset),
                               desc="CCR"):
                res = {
                    "dataset": dataset_name,
                    "model": config["name"],
                    "seed": seed,
                }
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        res[key] = float(value.cpu().numpy())
                    else:
                        res[key] = value
                search_result_ccr.append(res)

df = pd.DataFrame.from_records(search_result_eecr)
file_name = os.path.join("/home/loebbert/projects/thesis", "experiments",
                         "01_performance", "search_result_eecr_df.pkl")
df.to_pickle(file_name)

file_name = os.path.join("/home/loebbert/projects/thesis", "experiments",
                         "01_performance", "search_result_ccr_df.pkl")
with open(file_name, "wb") as file:
    pickle.dump(search_result_ccr, file)
df = pd.DataFrame.from_records(search_result_ccr)
df.to_pickle(file_name)