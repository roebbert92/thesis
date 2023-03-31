import pandas as pd
import json
from nltk.tokenize import sent_tokenize


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
