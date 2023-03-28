from dataclasses import dataclass, asdict
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize


@dataclass()
class Entity():
    type: str
    start: int
    end: int


def conll03_to_json(train_file: str, dev_file: str, test_file: str):
    conll03_types = {}

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
                if line == "-DOCSTART- -X- -X- O" or line == "":
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
                        if len(doc["tokens"]) > 60:
                            start = 0
                            for sent in sent_tokenize(" ".join(doc["tokens"])):
                                end = len(sent.split()) - 1 + start
                                current_doc = {
                                    "tokens":
                                    sent,
                                    "extended":
                                    sent,
                                    "entities": [
                                        ent for ent in doc["entities"]
                                        if start <= ent["start"] <= end
                                        and start <= ent["end"] <= end
                                    ]
                                }
                                dataset.append(current_doc)
                                start = end + 1
                        else:
                            doc["extended"] = doc["tokens"]
                            dataset.append(doc)
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
                        type = tag[2:]
                        types.append(type)
                        bio_label = tag[0]
                        if bio_label == "B":
                            # B = start entity + recognize type
                            current_types[type] = Entity(type, idx, idx)
                            conll03_types[type] = {"short": type}
                        elif bio_label == "I":
                            # I = current_type.end += 1
                            try:
                                assert type in current_types
                                current_types[type].end = idx
                            except AssertionError:
                                print("Annotation error: ", file_name, line_nr,
                                      items)
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
        dataset.append(doc)
        with open(f"{os.path.dirname(train_file)}/conll03_{name}.json",
                  "w",
                  encoding="utf-8") as json_file:
            json.dump(dataset, json_file)

    with open(f"{os.path.dirname(train_file)}/conll03_types.json",
              "w",
              encoding="utf-8") as json_file:
        json.dump({"entities": conll03_types}, json_file)
