import os
import sys
from typing import List

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

from search.utils import EMBEDDING_DIM, EMBEDDING_MODEL, get_sentences_from_documents
from haystack.document_stores import FAISSDocumentStore, BaseDocumentStore, ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack import Pipeline
from data_preparation.utils import remove_exact_matches
import json
import faiss


def create_sent_faiss_document_store():
    faiss_doc_store_path = os.path.join(thesis_path, "search", "sent",
                                        "faiss_document_store.db")
    if os.path.exists(faiss_doc_store_path):
        os.remove(faiss_doc_store_path)
    file = open(faiss_doc_store_path, "w")
    file.close()
    document_store = FAISSDocumentStore(
        #sql_url="postgresql://postgres:thesis123.@localhost:5432/sent",
        sql_url=f"sqlite:///{faiss_doc_store_path}",
        #faiss_index_factory_str="OPQ128_384,IVF20000,PQ128",
        embedding_dim=EMBEDDING_DIM,
        similarity="cosine")
    return document_store


def add_multiconer_sentences(doc_store: BaseDocumentStore):
    if doc_store.get_document_count() == 0:
        with open(os.path.join(thesis_path, "data", "multiconer",
                               "multiconer_test.json"),
                  "r",
                  encoding="utf-8") as file:
            multiconer = json.load(file)

        with open(os.path.join(thesis_path, "data/mlowner/lowner_train.json"),
                  encoding="utf-8") as file:
            lowner_train = json.load(file)
        with open(os.path.join(thesis_path, "data/mlowner/lowner_dev.json"),
                  encoding="utf-8") as file:
            lowner_dev = json.load(file)
        with open(os.path.join(thesis_path, "data/mlowner/lowner_test.json"),
                  encoding="utf-8") as file:
            lowner_test = json.load(file)

        filtered_multiconer = remove_exact_matches(
            multiconer, lowner_train + lowner_dev + lowner_test)
        documents = get_sentences_from_documents(filtered_multiconer)
        doc_store.write_documents(documents)


def train_update_sent_faiss_index(
        document_store: FAISSDocumentStore,  # type: ignore
        retriever: EmbeddingRetriever):
    document_store.update_embeddings(retriever)
    document_store.save(index_path=os.path.join(thesis_path, "search", "sent",
                                                "faiss_index.faiss"),
                        config_path=os.path.join(thesis_path, "search", "sent",
                                                 "faiss_config.json"))


def add_sent_search_components(search: Pipeline,
                               search_algorithm: str,
                               search_topk: int,
                               join_documents_input: List[str] = [],
                               reset=False):

    if search_algorithm == "bm25":
        document_store = ElasticsearchDocumentStore(
            index="sent",
            embedding_dim=EMBEDDING_DIM,
            similarity="cosine",
            recreate_index=reset)
        bm25_retriever = BM25Retriever(document_store, top_k=search_topk)
        add_multiconer_sentences(document_store)
        search.add_node(component=bm25_retriever,
                        name="SentBM25Retriever",
                        inputs=["Query"])
        join_documents_input.append("SentBM25Retriever")
    elif search_algorithm.startswith("ann"):
        faiss_index_path = os.path.join(thesis_path, "search", "sent",
                                        "faiss_index.faiss")
        if not os.path.exists(faiss_index_path) or reset:
            document_store = create_sent_faiss_document_store()
            ann_retriever = EmbeddingRetriever(
                document_store=document_store,  # type: ignore
                embedding_model=EMBEDDING_MODEL,
                model_format="sentence_transformers",
                top_k=search_topk)
            add_multiconer_sentences(document_store)  # type: ignore
            train_update_sent_faiss_index(document_store, ann_retriever)

        document_store = FAISSDocumentStore.load(  # type: ignore
            index_path=faiss_index_path,
            config_path=os.path.join(thesis_path, "search", "sent",
                                     "faiss_config.json"))
        document_store.faiss_indexes[
            document_store.index] = faiss.index_cpu_to_all_gpus(
                index=document_store.faiss_indexes[document_store.index])
        ann_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=search_topk)
        search.add_node(component=ann_retriever,
                        name="SentANNRetriever",
                        inputs=["Query"])
        join_documents_input.append("SentANNRetriever")

    if len(search.components) == 0:
        raise Exception(
            "Argument error: search_algorithm - must be: bm25 | ann , but is: "
            + search_algorithm)