from dataclasses import dataclass, asdict
import os
import json
from typing import List, Optional
from data_preparation import utils


@dataclass()
class Entity():
    type: str
    start: int
    end: int


def asp_dataset_to_asp_predictions(dataset: List[dict]):
    return {
        item["doc_id"]: [(ent["start"], ent["end"] - 1, ent["type"])
                         for ent in item["entities"]]
        for item in dataset
    }


def read_flair_predictions_to_asp(dataset_name: str, file_name: str):
    dataset = []
    idx, current_types, doc = -1, {}, {
        "tokens": [],  # list of tokens for the model to copy from
        "extended":
        [],  # list of input tokens. Prompts, instructions, etc. go here
        # list of dict:{"type": type, "start": start, "end": end}, format: [start, end)
        "entities": []
    }
    with open(file_name, encoding="utf-8") as file:
        for line_nr, line in enumerate(file):
            line = line.strip()
            if line == "":
                # clean up current_types
                for current_type in current_types.values():
                    try:
                        current_type.end = idx + 1
                        assert current_type.start < current_type.end
                        doc["entities"].append(asdict(current_type))
                    except AssertionError:
                        print("Annotation error: ", file_name, line_nr,
                              current_type)
                current_types = {}

                if doc is not None and len(doc["tokens"]) > 0:
                    if utils.is_supported_doc(doc["tokens"]):
                        doc["extended"] = doc["tokens"]
                        dataset.append(doc)
                    else:
                        print(file_name, line_nr, doc)
                doc = {"tokens": [], "extended": [], "entities": []}
                idx = -1
                continue
            else:
                idx += 1
                items = line.split()
                assert len(items) == 3, line

                token, _, bio_tag = items
                doc["tokens"].append(token)

                tags = bio_tag.split(",")
                types = []
                for tag in tags:
                    type = tag[2:]
                    types.append(type)
                    bio_label = tag[0]
                    if bio_label == "B":
                        # check if same type is already begun before
                        if type in current_types:
                            current_types[type].end = idx
                            assert current_types[type].start < current_types[
                                type].end
                            doc["entities"].append(asdict(current_types[type]))
                        # B = start entity + recognize type
                        current_types[type] = Entity(type, idx, idx)
                    elif bio_label == "I":
                        # I = current_type.end += 1
                        # According to paper there are entities that start with I
                        if type in current_types:
                            current_types[type].end = idx
                        else:
                            current_types[type] = Entity(type, idx, idx)
                    elif bio_label == "O":
                        # O = delete all types from current_types
                        for current_type in current_types.values():
                            try:
                                current_type.end = idx
                                assert current_type.start < current_type.end
                                doc["entities"].append(asdict(current_type))
                            except AssertionError:
                                print("Annotation error: ", file_name, line_nr,
                                      current_type, items)
                        current_types = {}
                # if types are not in tags -> entity ended
                types_not_in_tags = set(current_types).difference(set(types))
                for type in types_not_in_tags:
                    current_type = current_types[type]
                    try:
                        current_type.end = idx
                        assert current_type.start < current_type.end
                        doc["entities"].append(asdict(current_type))
                        del current_types[type]
                    except AssertionError:
                        print("Annotation error: ", file_name, line_nr,
                              current_type, items)
    if doc is not None and len(doc["tokens"]) > 0:
        doc["extended"] = doc["tokens"]
        dataset.append(doc)

    for doc_id, doc in enumerate(dataset):
        doc["doc_id"] = dataset_name + "_" + str(doc_id)

    return dataset
