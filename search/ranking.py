from typing import Any, Dict, List, Optional, Union
from haystack import BaseComponent, Document
import copy

from haystack.schema import Document, MultiLabel


class PrepareGazetteers(BaseComponent):
    outgoing_edges = 1

    def run(self, documents: List[Document]):
        docs = []
        for doc in documents:
            new_doc = copy.deepcopy(doc)
            if doc.meta["data_type"] == "gazetteers":
                new_doc.content = doc.meta["type"] + ": " + doc.content
            docs.append(new_doc)

        output = {"documents": docs}
        return output, "output_1"

    def run_batch(self, documents: List[List[Document]]):
        output = {"documents": []}
        for batch in documents:
            docs = []
            for doc in batch:
                new_doc = copy.deepcopy(doc)
                if doc.meta["data_type"] == "gazetteers":
                    new_doc.content = doc.meta["type"] + ": " + doc.content
                docs.append(new_doc)
            output["documents"].append(docs)
        return output, "output_1"


class PostprocessGazetteers(BaseComponent):
    outgoing_edges = 1

    def run(self, documents: List[Document]):
        docs = []
        for doc in documents:
            new_doc = copy.deepcopy(doc)
            if doc.meta["data_type"] == "gazetteers":
                new_doc.content = doc.content.replace(doc.meta["type"] + ": ",
                                                      "")
            docs.append(new_doc)

        output = {"documents": docs}
        return output, "output_1"

    def run_batch(self, documents: List[List[Document]]):
        output = {"documents": []}
        for batch in documents:
            docs = []
            for doc in batch:
                new_doc = copy.deepcopy(doc)
                if doc.meta["data_type"] == "gazetteers":
                    new_doc.content = doc.content.replace(
                        doc.meta["type"] + ": ", "")
                docs.append(new_doc)
            output["documents"].append(docs)
        return output, "output_1"