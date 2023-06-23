from collections import defaultdict
from haystack import Document
import uuid

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def get_gazetteers_from_documents(docs,
                                  name: str = "",
                                  prepend_type: bool = False):
    items = defaultdict(dict)
    for doc in docs:
        for entity in doc["entities"]:
            ne = " ".join(doc["tokens"][entity["start"]:entity["end"]])
            key = entity["type"] + "_" + ne
            if "doc_id" not in items[key]:
                items[key]["doc_id"] = []
            if doc["doc_id"] not in items[key]["doc_id"]:
                items[key]["doc_id"].append(doc["doc_id"])
            items[key]["type"] = entity["type"]
            items[key]["content"] = ne
    return [
        Document(
            id=str(uuid.uuid4()),
            content=doc["content"]
            if not prepend_type else f"{doc['type']}: {doc['content']}",
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
        if len(doc["entities"]) > 0:
            documents.append(
                Document(id=str(uuid.uuid4()),
                         content=" ".join(doc["extended"]),
                         meta={
                             "entities": doc["entities"],
                             "data_type": "sentences",
                             "doc_id": doc["doc_id"]
                         }))
    return documents