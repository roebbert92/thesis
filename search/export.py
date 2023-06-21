import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from search.utils import EMBEDDING_DIM

from haystack.document_stores import ElasticsearchDocumentStore
import pickle

gaz_doc_store = ElasticsearchDocumentStore(index="gaz",
                                           embedding_dim=EMBEDDING_DIM,
                                           similarity="cosine")
lownergaz_doc_store = ElasticsearchDocumentStore(index="lownergaz",
                                                 embedding_dim=EMBEDDING_DIM,
                                                 similarity="cosine")
sent_doc_store = ElasticsearchDocumentStore(index="sent",
                                            embedding_dim=EMBEDDING_DIM,
                                            similarity="cosine")

doc_stores = {
    "gaz": gaz_doc_store,
    "lownergaz": lownergaz_doc_store,
    "sent": sent_doc_store
}

for folder_name, doc_store in doc_stores.items():
    export_path = os.path.join(thesis_path, "search", folder_name,
                               "database_export.pkl")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "wb") as file:
        pickle.dump(doc_store.get_all_documents(), file)
