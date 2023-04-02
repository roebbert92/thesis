import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import pandas as pd
from data_preparation.wnut import wnut_to_json
from data.statistics_json import data_prep_check
import os
import numpy as np

dataset_files = {
    "wnut": {
        "train": "/home/loebbert/projects/thesis/data/wnut/wnut17train.conll",
        "dev": "/home/loebbert/projects/thesis/data/wnut/emerging.dev.conll",
        "test":
        "/home/loebbert/projects/thesis/data/wnut/emerging.test.annotated"
    }
}

methods = {"ours": wnut_to_json}

official_inputs = pd.DataFrame.from_records([
    {
        "Sentences": np.nan,
        "Tokens": np.nan,
        "Entities": np.nan
    },
    {
        "Sentences": 1008,
        "Tokens": 15734,
        "Entities": 836
    },
    {
        "Sentences": 1287,
        "Tokens": 23394,
        "Entities": 1079
    },
],
                                            index=dataset_files["wnut"].keys())
official_entities = pd.DataFrame.from_records(
    [{
        "Total": np.nan,
        "person": np.nan,
        "location": np.nan,
        "corporation": np.nan,
        "product": np.nan,
        "creative-work": np.nan,
        "group": np.nan
    }, {
        "Total": 836,
        "person": 470,
        "location": 74,
        "corporation": 34,
        "product": 114,
        "creative-work": 105,
        "group": 39
    }, {
        "Total": 1079,
        "person": 429,
        "location": 150,
        "corporation": 66,
        "product": 127,
        "creative-work": 142,
        "group": 165
    }],
    index=dataset_files["wnut"].keys())

data_prep_check("wnut", dataset_files, methods,
                "/home/loebbert/projects/thesis/experiments/wnut",
                official_inputs, official_entities)