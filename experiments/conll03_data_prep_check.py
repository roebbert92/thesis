import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import pandas as pd
from data_preparation.conll03 import conll03_to_json, asp_conll03_to_json
from data.statistics_json import data_prep_check
import os

dataset_files = {
    "huggingface": {
        "train":
        "/home/loebbert/projects/thesis/data/conll03/huggingface/train.txt",
        "dev":
        "/home/loebbert/projects/thesis/data/conll03/huggingface/valid.txt",
        "test":
        "/home/loebbert/projects/thesis/data/conll03/huggingface/test.txt"
    },
    "eth": {
        "train": "/home/loebbert/projects/thesis/data/conll03/eth/train.txt",
        "dev": "/home/loebbert/projects/thesis/data/conll03/eth/dev.txt",
        "test": "/home/loebbert/projects/thesis/data/conll03/eth/test.txt",
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
                                            index=dataset_files["eth"].keys())
official_entities = pd.DataFrame.from_records(
    [{
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
    index=dataset_files["eth"].keys())

data_prep_check("conll03", dataset_files, methods,
                "/home/loebbert/projects/thesis/experiments/conll03",
                official_inputs, official_entities)
