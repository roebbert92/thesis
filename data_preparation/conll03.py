from dataclasses import dataclass, asdict
import os
import json
from typing import Optional
from nltk.tokenize import sent_tokenize, word_tokenize


@dataclass()
class Entity():
    type: str
    start: int
    end: int


def conll03_to_json(train_file: str,
                    dev_file: str,
                    test_file: str,
                    dir_path: Optional[str] = None):
    if dir_path is None:
        dir_path = os.path.dirname(train_file)

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
                        # if len(doc["tokens"]) > 60:
                        #     start = 0
                        #     for sent in sent_tokenize(" ".join(doc["tokens"])):
                        #         end = len(sent.split()) - 1 + start
                        #         current_doc = {
                        #             "tokens":
                        #             sent,
                        #             "extended":
                        #             sent,
                        #             "entities": [
                        #                 ent for ent in doc["entities"]
                        #                 if start <= ent["start"] <= end
                        #                 and start <= ent["end"] <= end
                        #             ]
                        #         }
                        #         dataset.append(current_doc)
                        #         start = end + 1
                        # else:
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
                            # check if same type is already begun before
                            if type in current_types:
                                current_types[type].end = idx
                                assert current_types[
                                    type].start < current_types[type].end
                                doc["entities"].append(
                                    asdict(current_types[type]))
                            # B = start entity + recognize type
                            current_types[type] = Entity(type, idx, idx)
                            conll03_types[type] = {"short": type}
                        elif bio_label == "I":
                            # I = current_type.end += 1
                            # According to paper there are entities that start with I
                            if type in current_types:
                                current_types[type].end = idx
                            else:
                                current_types[type] = Entity(type, idx, idx)
                                conll03_types[type] = {"short": type}
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
            doc["doc_id"] = "conll03_" + name + "_" + str(doc_id)

        with open(f"{dir_path}/conll03_{name}.json", "w",
                  encoding="utf-8") as json_file:
            json.dump(dataset, json_file)

    with open(f"{dir_path}/conll03_types.json", "w",
              encoding="utf-8") as json_file:
        json.dump({"entities": conll03_types}, json_file)


def asp_conll03_to_json(train_file: str,
                        dev_file: str,
                        test_file: str,
                        dir_path: Optional[str] = None):
    if dir_path is None:
        dir_path = os.path.dirname(train_file)

    conll03_datasets, conll03_types = {}, {}

    files = {"train": train_file, "dev": dev_file, "test": test_file}

    for name, file_name in files.items():
        data = open(file_name).readlines()

        dataset = []
        idx, start, current_type, doc = -1, None, None, None
        for line in data:
            line = line.strip()

            if line == "-DOCSTART- -X- -X- O":  # new doc
                if doc is not None:
                    # when extended is not the same as tokens
                    # mark where to copy from with <extra_id_22> and <extra_id_23>
                    # E.g.
                    # Extract entities such as apple, orange, lemon <extra_id_22> Give me a mango . <extra_id_23>
                    # See ace05_to_json.py for example of extending the input

                    # FIX: missing entities
                    if start is not None:
                        doc['entities'].append({
                            "type":
                            current_type,
                            "start":
                            start,
                            "end":
                            idx if idx > start else idx + 1
                        })

                    doc["extended"] = doc["tokens"]
                    dataset.append(doc)
                doc = {
                    "tokens": [],  # list of tokens for the model to copy from
                    "extended":
                    [],  # list of input tokens. Prompts, instructions, etc. go here
                    "entities": [
                    ]  # list of dict:{"type": type, "start": start, "end": end}, format: [start, end)
                }
                idx, start = -1, None
                continue
            elif line == "":
                if len(doc["tokens"]) > 800 and name == "train":  # clip
                    if doc is not None:
                        doc["extended"] = doc["tokens"]
                        dataset.append(doc)
                    doc = {"tokens": [], "extended": [], "entities": []}
                    idx, start = -1, None
                    continue
                # new sentence
                pass
            else:
                idx += 1
                items = line.split()
                assert len(items) == 4, line

                token, _, _, bio_tag = items
                doc["tokens"].append(items[0])

                if bio_tag[0] == 'I':
                    if start is None:
                        start = idx
                        current_type = bio_tag[2:]
                        conll03_types[current_type] = {"short": current_type}
                elif bio_tag[0] == 'O':
                    if start is not None:
                        doc['entities'].append({
                            "type": current_type,
                            "start": start,
                            "end": idx
                        })
                    start = None
                elif bio_tag[0] == 'B':
                    if start is not None:
                        doc['entities'].append({
                            "type": current_type,
                            "start": start,
                            "end": idx
                        })
                    start = idx
                    current_type = bio_tag[2:]
                    conll03_types[current_type] = {"short": current_type}
        dataset.append(doc)
        conll03_datasets[name] = dataset
    for name in conll03_datasets:
        with open(f"{dir_path}/conll03_{name}.json", 'w') as fout:
            json.dump(conll03_datasets[name], fout)
    with open(f"{dir_path}/conll03_types.json", 'w') as fout:
        json.dump({"entities": conll03_types}, fout)