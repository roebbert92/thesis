from collections import defaultdict
from functools import reduce
import json
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_preprocessing.tokenize import query_database

from haystack import Pipeline, Document

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def get_or_filter_from_list(key_name, values):
    return {"$or": [{key_name: value} for value in values]}


def get_gazetteers_from_documents(docs, name: str = ""):
    items = defaultdict(dict)
    for doc in docs:
        for entity in doc["entities"]:
            ne = " ".join(doc["tokens"][entity["start"]:entity["end"]])
            dataset_part = name if len(name) > 0 else doc["doc_id"].split(
                "_")[1]
            key = dataset_part + "_" + entity["type"] + "_" + ne
            if "doc_id" not in items[key]:
                items[key]["doc_id"] = []
            if doc["doc_id"] not in items[key]["doc_id"]:
                items[key]["doc_id"].append(doc["doc_id"])
            if "dataset" not in items[key]:
                items[key]["dataset"] = []
            if dataset_part not in items[key]["dataset"]:
                items[key]["dataset"].append(dataset_part)
            items[key]["type"] = entity["type"]
            items[key]["content"] = ne
    return [
        Document(
            content=doc["content"],
            meta={
                #"doc_id": doc["doc_id"],
                #"dataset": doc["dataset"],
                "type": doc["type"],
                "data_type": "gazetteers"
            }) for doc in items.values()
    ]


def get_sentences_from_documents(docs, name: str = ""):
    documents = []
    for doc in docs:
        #dataset_part = name if len(name) > 0 else doc["doc_id"].split("_")[1]
        documents.append(
            Document(
                content=" ".join(doc["tokens"]),
                meta={
                    "entities": doc["entities"],
                    "data_type": "sentences",
                    #"doc_id": [doc["doc_id"]],
                    #"dataset": [dataset_part],
                }))
    return documents


def factors(n):
    return sorted(
        list(
            set(
                reduce(list.__add__,
                       ([i, n // i]
                        for i in range(1,
                                       int(n**0.5) + 1) if not n % i)))))


def get_search_results(search: Pipeline, file_name: str):
    with open(file_name, encoding="utf-8") as file:
        instances = json.load(file)
    results = {
        instance_idx: result
        for instance_idx, result in query_database(instances, search)
    }
    return results
