import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import pandas as pd
from data_preparation.conll03 import conll03_to_json, asp_conll03_to_json
from data.statistics_json import create_dataset_stats
import os

eth_files = {
    "train": "/home/loebbert/projects/thesis/data/conll03/asp/train.txt",
    "dev": "/home/loebbert/projects/thesis/data/conll03/asp/dev.txt",
    "test": "/home/loebbert/projects/thesis/data/conll03/asp/test.txt",
}

huggingface_files = {
    "train":
    "/home/loebbert/projects/thesis/data/conll03/huggingface/train.txt",
    "dev": "/home/loebbert/projects/thesis/data/conll03/huggingface/valid.txt",
    "test": "/home/loebbert/projects/thesis/data/conll03/huggingface/test.txt"
}

conll03_files = {
    "huggingface": {
        "train":
        "/home/loebbert/projects/thesis/data/conll03/huggingface/train.txt",
        "dev":
        "/home/loebbert/projects/thesis/data/conll03/huggingface/valid.txt",
        "test":
        "/home/loebbert/projects/thesis/data/conll03/huggingface/test.txt"
    },
    "eth": {
        "train": "/home/loebbert/projects/thesis/data/conll03/asp/train.txt",
        "dev": "/home/loebbert/projects/thesis/data/conll03/asp/dev.txt",
        "test": "/home/loebbert/projects/thesis/data/conll03/asp/test.txt",
    }
}

methods = {"ours": conll03_to_json, "asp": asp_conll03_to_json}

official_inputs = pd.DataFrame.from_records([
    {
        "Sentences": 14987,
        "Tokens": 203621,
        "Entities": 7140 + 3438 + 6321 + 6600
    },
    {
        "Sentences": 3466,
        "Tokens": 51362,
        "Entities": 1837 + 922 + 1341 + 1842
    },
    {
        "Sentences": 3684,
        "Tokens": 46435,
        "Entities": 1668 + 702 + 1661 + 1617
    },
],
                                            index=huggingface_files.keys())
official_entities = pd.DataFrame.from_records([{
    "Total": 7140 + 3438 + 6321 + 6600,
    "LOC": 7140,
    "MISC": 3438,
    "ORG": 6321,
    "PER": 6600
}, {
    "Total": 1837 + 922 + 1341 + 1842,
    "LOC": 1837,
    "MISC": 922,
    "ORG": 1341,
    "PER": 1842
}, {
    "Total": 1668 + 702 + 1661 + 1617,
    "LOC": 1668,
    "MISC": 702,
    "ORG": 1661,
    "PER": 1617
}],
                                              index=huggingface_files.keys())

sentences = []
tokens = []
entities = []
total_entities = []
loc_entities = []
misc_entities = []
org_entities = []
per_entities = []


def append_to_lists(stats_input, stats_entities, data_name, method_name):
    # sentence
    sentence = stats_input["Sentences"].copy()
    sentence.name = data_name + "_" + method_name
    sentences.append(sentence)
    # token
    token = stats_input["Tokens"].copy()
    token.name = data_name + "_" + method_name
    tokens.append(token)
    # entity
    entity = stats_input["Entities"].copy()
    entity.name = data_name + "_" + method_name
    entities.append(entity)
    # total entities
    total_entity = stats_entities["Total"].copy()
    total_entity.name = data_name + "_" + method_name
    total_entities.append(total_entity)
    # loc
    loc = stats_entities["LOC"].copy()
    loc.name = data_name + "_" + method_name
    loc_entities.append(loc)
    # misc
    misc = stats_entities["MISC"].copy()
    misc.name = data_name + "_" + method_name
    misc_entities.append(misc)
    # org
    org = stats_entities["ORG"].copy()
    org.name = data_name + "_" + method_name
    org_entities.append(org)
    # per
    per = stats_entities["PER"].copy()
    per.name = data_name + "_" + method_name
    per_entities.append(per)


append_to_lists(official_inputs, official_entities, "original", "paper")

# for each method run all files
for method_name, method in methods.items():
    for data_name, files in conll03_files.items():
        dir_path = os.path.join(os.path.dirname(files["train"]), method_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        method(files["train"], files["dev"], files["test"], dir_path=dir_path)
        stats_input, stats_entities = create_dataset_stats(
            os.path.join(dir_path, "conll03_train.json"),
            os.path.join(dir_path, "conll03_dev.json"),
            os.path.join(dir_path, "conll03_test.json"),
        )
        append_to_lists(stats_input, stats_entities, data_name, method_name)


def save_to_csv(dir_path, name, data_list):
    df = pd.concat(data_list, axis=1)
    df.to_csv(os.path.join(dir_path, name + ".csv"))


save_to_csv("/home/loebbert/projects/thesis/experiments/conll03", "sentences",
            sentences)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03", "tokens",
            tokens)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03", "entities",
            entities)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03",
            "total_entities", total_entities)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03",
            "loc_entities", loc_entities)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03",
            "misc_entities", misc_entities)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03",
            "org_entities", org_entities)
save_to_csv("/home/loebbert/projects/thesis/experiments/conll03",
            "per_entities", per_entities)
