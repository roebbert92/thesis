import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import pickle
import json
from configs.asp_t5 import T5_ASP_LOWNERGAZ_SENT, T5_ASP_LOWNERGAZ, T5_ASP_GAZ_SENT, T5_ASP_GAZ, T5_ASP_SENT
from hyperparameter_tuning.t5_asp_lownergaz_sent import setup_database as setup_database_lownergaz_sent
from hyperparameter_tuning.t5_asp_gaz_sent import setup_database as setup_database_gaz_sent
from hyperparameter_tuning.t5_asp_lownergaz import setup_database as setup_database_lownergaz
from hyperparameter_tuning.t5_asp_gaz import setup_database as setup_database_gaz
from hyperparameter_tuning.t5_asp_sent import setup_database as setup_database_sent
from hyperparameter_tuning.utils import get_search_results

configs = [
    T5_ASP_LOWNERGAZ_SENT, T5_ASP_LOWNERGAZ, T5_ASP_GAZ_SENT, T5_ASP_GAZ,
    T5_ASP_SENT
]

files = {
    "train": os.path.join(thesis_path, "data", "wnut", "wnut_train.json"),
    "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
    "test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
}

with open(files["train"], encoding="utf-8") as file:
    lowner_train = json.load(file)

with open(files["dev"], encoding="utf-8") as file:
    lowner_dev = json.load(file)

with open(files["test"], encoding="utf-8") as file:
    lowner_test = json.load(file)

parts = ["train", "dev", "test"]
datasets = {"train": lowner_train, "dev": lowner_dev, "test": lowner_test}

for config in configs:
    config.update({
        "data_path":
        os.path.join(thesis_path, "experiments", "01_performance", "data")
    })

for config in configs:
    search = None
    if config["name"] == "t5_asp_lownergaz_sent":
        search = setup_database_lownergaz_sent(
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
    if config["name"] == "t5_asp_lownergaz":
        search = setup_database_lownergaz(config["search_algorithm"],
                                          config["search_topk"])
    if config["name"] == "t5_asp_gaz":
        search = setup_database_gaz(config["search_algorithm"],
                                    config["search_topk"])
    if config["name"] == "t5_asp_sent":
        search = setup_database_sent(config["search_algorithm"],
                                     config["search_topk"])

    # go through all datasets
    for part in parts:
        search_results = {}
        dataset = datasets[part]
        dataset_name = "wnut_" + part
        # save search results for augmentation
        file_name = os.path.join(config["data_path"], "01_search_results",
                                 config['name'], f"{dataset_name}.pkl")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        # get search results
        search_result = None
        if search is not None:
            if not os.path.exists(file_name):
                search_result = get_search_results(search, dataset)
                with open(file_name, "wb") as file:
                    pickle.dump(search_result, file)
            else:
                with open(file_name, "rb") as file:
                    search_result = pickle.load(file)

            search_results[part] = search_result

# Gaz

# Sent

# LownerGaz

# Gaz+Sent

# LownerGaz+Sent