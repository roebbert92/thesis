import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from search.utils import EMBEDDING_DIM

from haystack.document_stores import ElasticsearchDocumentStore
import json

gaz_doc_store = ElasticsearchDocumentStore(index="gaz",
                                           embedding_dim=EMBEDDING_DIM,
                                           similarity="cosine")
lownergaz_doc_store = ElasticsearchDocumentStore(index="lownergaz",
                                                 embedding_dim=EMBEDDING_DIM,
                                                 similarity="cosine")
sent_doc_store = ElasticsearchDocumentStore(index="sent",
                                            embedding_dim=EMBEDDING_DIM,
                                            similarity="cosine")
lownergaz_gaz_doc_store = ElasticsearchDocumentStore(
    index="lownergaz_gaz", embedding_dim=EMBEDDING_DIM, similarity="cosine")

doc_stores = {
    #"gaz": gaz_doc_store,
    #"lownergaz": lownergaz_doc_store,
    #"sent": sent_doc_store,
    "lownergaz_gaz": lownergaz_gaz_doc_store
}

for folder_name, doc_store in doc_stores.items():
    if folder_name in ["gaz", "sent"]:
        export_path = os.path.join(thesis_path, "data", "multiconer",
                                   f"multiconer_{folder_name}.json")
    elif folder_name == "lownergaz":
        export_path = os.path.join(thesis_path, "data", "mlowner",
                                   "lowner_gazetteer_db.json")
    else:
        export_path = os.path.join(thesis_path, "data", "mlowner",
                                   "lownergaz_gaz.json")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as file:
        json.dump([{
            "content": str(doc.content),
            "meta": doc.meta
        } for doc in doc_store.get_all_documents()], file)
