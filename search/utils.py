import copy
import sys
import os

thesis_path = "/" + os.path.join(*os.getcwd().split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from collections import defaultdict
from haystack import Document
import uuid

from data_preparation.utils import is_supported_doc

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def get_gazetteers_from_documents(docs, name: str = "", prepend_type: bool = False):
    if "entities" in docs[0]:
        return get_sentences_from_documents(docs, name)
    elif "entity" in docs[0]:
        documents = []
        for gaz in docs:
            if is_supported_doc(gaz["entity"].split()):
                entity = gaz["entity"]
                if prepend_type:
                    entity = f"{gaz['type']}: {gaz['entity']}"
                documents.append(
                    Document(
                        id=str(uuid.uuid4()),
                        content=entity,
                        meta={
                            "data_type": "gazetteers",
                            "type": gaz["type"],
                            "entity_id": gaz["entity_id"],
                        },
                    )
                )
        return documents


def get_sentences_from_documents(docs, name: str = ""):
    documents = []
    for doc in docs:
        # if len(doc["entities"]) > 0:
        entities = []
        for entity in doc["entities"]:
            if "error" not in entity or entity["error"] != 2:
                entities.append(entity)

        documents.append(
            Document(
                id=str(uuid.uuid4()),
                content=" ".join(doc["extended"]),
                meta={
                    "entities": entities,
                    "data_type": "sentences",
                    "doc_id": doc["doc_id"],
                },
            )
        )
    return documents
