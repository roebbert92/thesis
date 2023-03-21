from dataclasses import dataclass, asdict
import os
import json


@dataclass()
class Entity():
    type: str
    start: int
    end: int


def find_type(type_list: list, type_value: str):
    for item in type_list:
        if item.type == type_value:
            return item
    return None


def wnut_to_json(train_file: str, dev_file: str, test_file: str):
    wnut_types = {}

    for file_name in [train_file, dev_file, test_file]:
        dataset = []
        idx, current_types, doc = -1, {}, {
            "tokens": [],  # list of tokens for the model to copy from
            "extended": [],  # list of input tokens. Prompts, instructions, etc. go here
            # list of dict:{"type": type, "start": start, "end": end}, format: [start, end)
            "entities": []
        }
        with open(file_name, encoding="utf-8") as file:
            for line_nr, line in enumerate(file):
                line = line.strip()
                if line == "":
                    if doc is not None:
                        doc["extended"] = doc["tokens"]
                        dataset.append(doc)
                    doc = {
                        "tokens": [],
                        "extended": [],
                        "entities": []
                    }
                    idx = -1
                    continue
                else:
                    idx += 1
                    items = line.split()
                    assert len(items) == 2, line

                    token, bio_tag = items
                    doc["tokens"].append(token)

                    tags = bio_tag.split(",")
                    types = []
                    for tag in tags:
                        type = tag[2:]
                        types.append(type)
                        bio_label = tag[0]
                        if bio_label == "B":
                            # B = start entity + recognize type
                            current_types[type] = Entity(type, idx, idx)
                            wnut_types[type] = {
                                "short": type
                            }
                        elif bio_label == "I":
                            # I = current_type.end += 1
                            try:
                                assert type in current_types
                                current_types[type].end = idx
                            except AssertionError:
                                print("Annotation error: ",
                                      file_name, line_nr, items)
                        elif bio_label == "O":
                            # O = delete all types from current_types
                            for current_type in current_types.values():
                                current_type.end = idx
                                doc["entities"].append(asdict(current_type))
                            current_types = {}
                    types_not_in_tags = set(
                        current_types).difference(set(types))
                    for type in types_not_in_tags:
                        current_type = current_types[type]
                        current_type.end = idx
                        doc["entities"].append(asdict(current_type))
                        del current_types[type]
        dataset.append(doc)
        with open(f"{os.path.splitext(file_name)[0]}.json", "w", encoding="utf-8") as json_file:
            json.dump(dataset, json_file)

    with open(f"{os.path.dirname(train_file)}/wnut_types.json", "w", encoding="utf-8") as json_file:
        json.dump({"entities": wnut_types}, json_file)
