import json
import sys
import os
from typing import Optional

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1]
)
sys.path.append(thesis_path)


from search.utils import get_gazetteers_from_documents


from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore, BaseDocumentStore
from haystack.nodes import BM25Retriever
from elasticsearch import Elasticsearch

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def add_unseen_multiconer_gazetteers(document_store: BaseDocumentStore):
    with open(
        os.path.join(thesis_path, "data", "multiconer", "multiconer_test.json"),
        "r",
        encoding="utf-8",
    ) as file:
        multiconer = json.load(file)
    # don't filter multiconer as the sentences are the gazetteers
    documents = get_gazetteers_from_documents(multiconer, prepend_type=False)
    gaz_docs = {f"{doc.content}_{doc.meta['type']}": doc.id for doc in documents}
    existing_docs = set()
    for doc in document_store.get_all_documents_generator():
        key = f"{doc.content}_{doc.meta['type']}"
        if key in gaz_docs:
            existing_docs.add(key)
    filtered_docs = [
        doc
        for doc in documents
        if f"{doc.content}_{doc.meta['type']}" not in existing_docs
    ]
    document_store.write_documents(filtered_docs)


def setup_database(
    search_algorithm: str, search_topk: int, reset=False, name: Optional[str] = None
):
    search = Pipeline()
    elasticsearch_client = Elasticsearch("http://localhost:9200")
    if not elasticsearch_client.indices.exists(index="lownergaz_gaz"):
        elasticsearch_client.indices.put_settings(
            index="lownergaz", body={"index": {"blocks": {"write": True}}}
        )
        elasticsearch_client.indices.clone(index="lownergaz", target="lownergaz_gaz")
        elasticsearch_client.indices.put_settings(
            index="lownergaz_gaz", body={"index": {"blocks": {"write": False}}}
        )
        document_store = ElasticsearchDocumentStore(
            index="lownergaz_gaz", embedding_dim=EMBEDDING_DIM, recreate_index=reset
        )
        bm25_retriever = BM25Retriever(document_store, top_k=search_topk)
        # add_lownergaz_gazetteers(document_store, [])
        add_unseen_multiconer_gazetteers(document_store)
        search.add_node(
            component=bm25_retriever, name="LownerGazBM25Retriever", inputs=["Query"]
        )
    else:
        document_store = ElasticsearchDocumentStore(
            index="lownergaz_gaz", embedding_dim=EMBEDDING_DIM, recreate_index=reset
        )
        bm25_retriever = BM25Retriever(document_store, top_k=search_topk)
        search.add_node(
            component=bm25_retriever, name="LownerGazBM25Retriever", inputs=["Query"]
        )

    return search


if __name__ == "__main__":
    setup_database("bm25", 10)
