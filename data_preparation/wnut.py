from dataclasses import dataclass, asdict
import os
import json
from typing import Optional


@dataclass()
class Entity():
    type: str
    start: int
    end: int


def wnut_to_json(train_file: str,
                 dev_file: str,
                 test_file: str,
                 dir_path: Optional[str] = None):
    if dir_path is None:
        dir_path = os.path.dirname(train_file)

    wnut_types = {}
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
                        doc["extended"] = doc["tokens"]
                        dataset.append(doc)
                    doc = {"tokens": [], "extended": [], "entities": []}
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
                            # check if same type is already begun before
                            if type in current_types:
                                current_types[type].end = idx
                                assert current_types[
                                    type].start < current_types[type].end
                                doc["entities"].append(
                                    asdict(current_types[type]))
                            # B = start entity + recognize type
                            current_types[type] = Entity(type, idx, idx)
                            wnut_types[type] = {"short": type}
                        elif bio_label == "I":
                            # I = current_type.end += 1
                            # According to paper there are entities that start with I
                            if type in current_types:
                                current_types[type].end = idx
                            else:
                                current_types[type] = Entity(type, idx, idx)
                                wnut_types[type] = {"short": type}
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
        with open(f"{dir_path}/wnut_{name}.json", "w",
                  encoding="utf-8") as json_file:
            json.dump(dataset, json_file)

    with open(f"{dir_path}/wnut_types.json", "w",
              encoding="utf-8") as json_file:
        json.dump({"entities": wnut_types}, json_file)
