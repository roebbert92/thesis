import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_preparation.utils import json_to_bio, json_to_bmes
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
import json
from search.utils import EMBEDDING_DIM


def to_bio(paths: dict):
    for part, path in list(paths.items()):
        paths[part] = json_to_bio(path)
    return paths


def to_bmes(paths: dict):
    for part, path in list(paths.items()):
        paths[part] = json_to_bmes(path)
    return paths


def get_lowner_bmes():
    paths = {
        "train": os.path.join(thesis_path, "data", "mlowner",
                              "lowner_train.json"),
        "dev": os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
        "test": os.path.join(thesis_path, "data", "mlowner",
                             "lowner_test.json"),
    }
    return to_bmes(paths)


def get_wnut_bmes():
    paths = {
        "train": os.path.join(thesis_path, "data", "wnut", "wnut_train.json"),
        "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
        "test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
    }
    return to_bmes(paths)


def get_multiconer_test_bmes():
    # get export from sent
    output_path = os.path.join(thesis_path, "data", "multiconer",
                               "multiconer_sent.json")
    if not os.path.exists(output_path):
        doc_store = ElasticsearchDocumentStore(index="sent",
                                               embedding_dim=EMBEDDING_DIM)

        items = [{
            "tokens": str(doc.content).split(" "),
            "entities": doc.meta["entities"]
        } for doc in doc_store.get_all_documents()]

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(items, file)

    return to_bmes({"test": output_path})


def get_lowner_bio():
    paths = {
        "train": os.path.join(thesis_path, "data", "mlowner",
                              "lowner_train.json"),
        "dev": os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
        "test": os.path.join(thesis_path, "data", "mlowner",
                             "lowner_test.json"),
    }
    return to_bio(paths)


def get_wnut_bio():
    paths = {
        "train": os.path.join(thesis_path, "data", "wnut", "wnut_train.json"),
        "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
        "test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
    }
    return to_bio(paths)


def get_multiconer_test_bio():
    # get export from sent
    output_path = os.path.join(thesis_path, "data", "multiconer",
                               "multiconer_sent.json")
    if not os.path.exists(output_path):
        doc_store = ElasticsearchDocumentStore(index="sent",
                                               embedding_dim=EMBEDDING_DIM)

        items = [{
            "tokens": str(doc.content).split(" "),
            "entities": doc.meta["entities"]
        } for doc in doc_store.get_all_documents()]

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(items, file)

    return to_bio({"test": output_path})


if __name__ == "__main__":
    get_wnut_bio()
    get_lowner_bio()
    get_lowner_bmes()
    get_wnut_bmes()