from dataclasses import dataclass, asdict
import os
import json
import regex
from typing import Optional
from data_preparation import utils


@dataclass()
class Entity():
    type: str
    start: int
    end: int


wnut_types = {
    "PER": "person",
    "GRP": "group",
    "CORP": "corporation",
    "LOC": "location",
    "CW": "creative-work",
    "PROD": "product"
}

pattern = regex.compile(r"\\(\w+)\"")


def gazetteer_entry_correction(entity: str):
    if "\t" in entity:
        entity = entity.replace("\t", "")
        if entity.endswith('"'):
            entity = entity[:-1]
            entity = pattern.sub('"\\1"', entity)
    return " ".join([token for token in entity.split(" ") if token])


def lowner_gaz_to_json(gaz_file: str):
    dir_path = os.path.dirname(gaz_file)

    gazetteer = []
    with open(gaz_file, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            try:
                items = line.split("\t")
                if len(items) < 4:
                    items = line.split(",")
                elif len(items) > 4:
                    items = []
                    start = 0
                    while len(items) < 3:
                        split = line.find("\t", start)
                        items.append(line[start:split])
                        start = split + 1
                    items.append(line[start:])
                assert len(items) == 4
                ent_id, ent_type, _, ent = items
                assert ent_type in wnut_types
                ent = gazetteer_entry_correction(ent)
                gazetteer.append({
                    "entity_id": ent_id,
                    "type": wnut_types[ent_type],
                    "entity": ent
                })
            except AssertionError:
                print(line)

    with open(f"{dir_path}/lowner_gazetteer.json", "w",
              encoding="utf-8") as json_file:
        json.dump(gazetteer, json_file)


def lowner_to_json(train_file: str,
                   dev_file: str,
                   test_file: str,
                   dir_path: Optional[str] = None):
    if dir_path is None:
        dir_path = os.path.dirname(train_file)

    files = {"train": train_file, "dev": dev_file, "test": test_file}

    for name, file_name in files.items():
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
                    assert len(items) == 4, line

                    token, _, _, bio_tag = items
                    doc["tokens"].append(token)

                    tags = bio_tag.split(",")
                    types = []
                    for tag in tags:
                        type = wnut_types[
                            tag[2:]] if not tag.startswith("O") else ""
                        types.append(type)
                        bio_label = tag[0]
                        if bio_label == "B":
                            # check if same type is already begun before
                            if type in current_types:
                                current_types[type].end = idx
                                assert current_types[
                                    type].start < current_types[type].end
                                doc["entities"].append(
                                    asdict(current_types[type]))
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
                                    doc["entities"].append(
                                        asdict(current_type))
                                except AssertionError:
                                    print("Annotation error: ", file_name,
                                          line_nr, current_type, items)
                            current_types = {}
                    # if types are not in tags -> entity ended
                    types_not_in_tags = set(current_types).difference(
                        set(types))
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
            doc["doc_id"] = "lowner_" + name + "_" + str(doc_id)

        with open(f"{dir_path}/lowner_{name}.json", "w",
                  encoding="utf-8") as json_file:
            json.dump(dataset, json_file)

    with open(f"{dir_path}/lowner_types.json", "w",
              encoding="utf-8") as json_file:
        json.dump(
            {
                "entities": {
                    "location": {
                        "short": "location"
                    },
                    "group": {
                        "short": "group"
                    },
                    "corporation": {
                        "short": "corporation"
                    },
                    "person": {
                        "short": "person"
                    },
                    "creative-work": {
                        "short": "creative-work"
                    },
                    "product": {
                        "short": "product"
                    }
                }
            }, json_file)
