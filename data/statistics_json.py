import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import pandas as pd
import json
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer
from collections import defaultdict
from typing import Optional

from data_preprocessing.tokenize import tokenize_json


def create_dataset_stats(train_json, dev_json, test_json):

    inputs = []
    entities = []

    files = {
        "train": train_json,
        "dev": dev_json,
        "test": test_json,
    }

    for part, filename in files.items():
        with open(filename, encoding="utf-8") as file:
            items = json.load(file)
        for item in items:
            inputs.append({
                "part":
                part,
                "tokens_len":
                len(item["tokens"]),
                "tokens_sent_count":
                len(sent_tokenize(" ".join(item["tokens"]))),
                "extended_len":
                len(item["extended"]),
                "extended_sent_count":
                len(sent_tokenize(" ".join(item["extended"]))),
                "entities_count":
                len(item["entities"])
            })
            for ent in item["entities"]:
                entities.append({
                    "part": part,
                    ent["type"]: 1,
                    "entity_len": ent["end"] - ent["start"]
                })

    df_inputs = pd.DataFrame.from_records(inputs)
    df_entities = pd.DataFrame.from_records(entities)

    summary_inputs = []
    summary_entities = []

    for part in files:
        summed_inputs = pd.DataFrame.from_records([
            df_inputs[df_inputs.part == part][[
                "tokens_len", "tokens_sent_count", "extended_len",
                "extended_sent_count", "entities_count"
            ]].agg("sum").to_dict()
        ])
        summed_inputs.index = ["sum"]
        summary_inputs.append({
            "Sentences":
            summed_inputs["tokens_sent_count"].loc["sum"],
            "Tokens":
            summed_inputs["tokens_len"].loc["sum"],
            "Entities":
            summed_inputs["entities_count"].loc["sum"],
        })
        summed_entities = pd.DataFrame.from_records([
            df_entities[df_entities.part == part][[
                column for column in df_entities.columns if column != "part"
            ]].agg("sum").to_dict()
        ])
        summed_entities.index = ["sum"]
        ents = {"Total": len(df_entities[df_entities.part == part].index)}
        for column in sorted(list(summed_entities.columns)):
            if column != "entity_len":
                ents[column] = summed_entities[column].loc["sum"]
        summary_entities.append(ents)

    return pd.DataFrame.from_records(
        summary_inputs,
        index=files.keys()), pd.DataFrame.from_records(summary_entities,
                                                       index=files.keys())


def create_tokenized_stats(train_jsonlines, dev_jsonlines, test_jsonlines):

    files = {
        "train": train_jsonlines,
        "dev": dev_jsonlines,
        "test": test_jsonlines
    }

    tokenized_inputs = []
    tokenized_targets = []

    for part, filename in files.items():
        with open(filename, encoding="utf-8") as file:
            items = json.load(file)
        for item in items:
            tokenized_inputs.append({
                "part":
                part,
                "Tokenized Input":
                len(item["input_sentence"])
            })
            tokenized_targets.append({
                "part":
                part,
                "Tokenized Target":
                len(item["target_sentence"])
            })

    df_inputs = pd.DataFrame.from_records(tokenized_inputs)
    df_targets = pd.DataFrame.from_records(tokenized_targets)

    inputs_summary = []
    targets_summary = []

    for part in files:
        inputs_summary.extend(df_inputs[df_inputs.part == part][[
            column for column in df_inputs.columns if column != "part"
        ]].describe().T.to_dict(orient="records"))
        targets_summary.extend(df_targets[df_targets.part == part][[
            column for column in df_targets.columns if column != "part"
        ]].describe().T.to_dict(orient="records"))

    return pd.DataFrame.from_records(
        inputs_summary,
        index=files.keys()), pd.DataFrame.from_records(targets_summary,
                                                       index=files.keys())


def data_prep_check(dataset_name: str,
                    dataset_files: dict,
                    methods: dict,
                    output_dir_path: str,
                    official_inputs: Optional[pd.DataFrame] = None,
                    official_entities: Optional[pd.DataFrame] = None):
    summaries = defaultdict(list)

    def append_to_lists(stats_input, stats_entities, stats_tokenized_inputs,
                        stats_tokenized_targets, data_name, method_name):

        for column in stats_input.columns:
            name = str(column).lower()
            series = stats_input[column].copy()
            series.name = data_name + "_" + method_name
            summaries[name].append(series)

        for column in stats_entities.columns:
            name = str(column).lower() + "_entities"
            series = stats_entities[column].copy()
            series.name = data_name + "_" + method_name
            summaries[name].append(series)

        if stats_tokenized_inputs is not None:
            # tokenized inputs
            name = "tokenized_inputs"
            for measure in ["count", "mean", "max"]:
                tokenized_input = stats_tokenized_inputs[measure].copy()
                tokenized_input.name = data_name + "_" + method_name + "_" + measure
                summaries[name].append(tokenized_input)

        if stats_tokenized_targets is not None:
            # tokenized targets
            name = "tokenized_targets"
            for measure in ["count", "mean", "max"]:
                tokenized_target = stats_tokenized_targets[measure].copy()
                tokenized_target.name = data_name + "_" + method_name + "_" + measure
                summaries[name].append(tokenized_target)

    if official_inputs is not None and official_entities is not None:
        append_to_lists(official_inputs, official_entities, None, None,
                        "original", "paper")

    # for each method run all files
    for method_name, method in methods.items():
        for data_name, files in dataset_files.items():
            # data preparation
            dir_path = os.path.join(os.path.dirname(files["train"]),
                                    method_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            method(files["train"],
                   files["dev"],
                   files["test"],
                   dir_path=dir_path)
            stats_input, stats_entities = create_dataset_stats(
                os.path.join(dir_path, dataset_name + "_train.json"),
                os.path.join(dir_path, dataset_name + "_dev.json"),
                os.path.join(dir_path, dataset_name + "_test.json"),
            )
            # tokenizing
            tokenizer = T5Tokenizer.from_pretrained("t5-small",
                                                    model_max_length=4096)
            tokenize_json(tokenizer,
                          os.path.join(dir_path, dataset_name + "_train.json"),
                          os.path.join(dir_path, dataset_name + "_dev.json"),
                          os.path.join(dir_path, dataset_name + "_test.json"),
                          os.path.join(dir_path, dataset_name + "_types.json"))
            stats_tokenized_inputs, stats_tokenized_targets = create_tokenized_stats(
                os.path.join(dir_path,
                             dataset_name + "_train.t5-small.jsonlines"),
                os.path.join(dir_path,
                             dataset_name + "_dev.t5-small.jsonlines"),
                os.path.join(dir_path,
                             dataset_name + "_test.t5-small.jsonlines"),
            )

            append_to_lists(stats_input, stats_entities,
                            stats_tokenized_inputs, stats_tokenized_targets,
                            data_name, method_name)

    for name, data_list in summaries.items():
        df = pd.concat(data_list, axis=1)
        df.to_csv(os.path.join(output_dir_path, name + ".csv"))