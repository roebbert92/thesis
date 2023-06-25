import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_preparation.utils import json_to_bio, convert_bio_to_bmes, to_bio
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
import json
from search.utils import EMBEDDING_DIM


def to_bmes(paths: dict):
    for part, path in list(paths.items()):
        bio_path = json_to_bio(path)
        paths[part] = convert_bio_to_bmes(bio_path)
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
