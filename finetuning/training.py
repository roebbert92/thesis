from collections import defaultdict
import copy
from functools import reduce
import json
import shutil
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from models.asp_t5 import ASPT5Model, get_tokenizer
from configs.asp_t5 import T5_BASE
from data_preprocessing.tensorize import NERDataProcessor, ner_collate_fn, NERCollator
from data_preprocessing.tokenize import tokenize_json, tokenize_database_json

from torch.utils.data import DataLoader
import torch
from finetuning.ray_logging import TuneReportCallback
from haystack import Pipeline, Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever, SentenceTransformersRanker, BM25Retriever
import lightning.fabric.utilities.seed as pl_seed
import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def setup_database(database_name: str, search_algorithm: str,
                   search_topk: int):
    document_store = ElasticsearchDocumentStore(
        index=database_name,
        embedding_dim=EMBEDDING_DIM,
        return_embedding=True if search_algorithm != "bm25" else False,
        recreate_index=True)
    search = Pipeline()
    if search_algorithm == "bm25":
        bm25_retriever = BM25Retriever(document_store, top_k=search_topk)
        search.add_node(component=bm25_retriever,
                        name="BM25Retriever",
                        inputs=["Query"])
    elif search_algorithm.startswith("ann"):
        ann_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=search_topk *
            2 if search_algorithm.endswith("ranking") else search_topk)
        search.add_node(component=ann_retriever,
                        name="ANNRetriever",
                        inputs=["Query"])
        if search_algorithm.endswith("ranking"):
            ranker = SentenceTransformersRanker(
                model_name_or_path=
                "sentence-transformers/msmarco-bert-base-dot-v5",
                top_k=search_topk)
            search.add_node(component=ranker,
                            name="Ranker",
                            inputs=["ANNRetriever"])

    if len(search.components) == 0:
        raise Exception(
            "Argument error: search_algorithm - must be: bm25 | ann | ann+ranking, but is: "
            + search_algorithm)

    return document_store, search


def get_or_filter_from_list(key_name, values):
    return {"$or": [{key_name: value} for value in values]}


def augment_dataset(tokenizer,
                    files,
                    filters,
                    database,
                    use_labels,
                    use_mentions,
                    prepend_search_results,
                    filter_exact_match,
                    filter_same_document,
                    filtered_document_ids={}):
    for part, dataset_filters in filters.items():
        tokenized_name = "tokenized_" + part
        files[tokenized_name], _, _ = tokenize_database_json(
            tokenizer,
            files[part],
            files["types"],
            database,
            None, {},
            use_labels,
            use_mentions,
            part,
            filters=get_or_filter_from_list("dataset", dataset_filters),
            filter_exact_match=filter_exact_match,
            filter_same_document=filter_same_document,
            filtered_document_ids=filtered_document_ids[part]
            if part in filtered_document_ids else [],
            prepend_search_results=prepend_search_results)


def get_documents_from_gazetteers(docs):
    items = defaultdict(dict)
    for doc in docs:
        for entity in doc["entities"]:
            ne = " ".join(doc["tokens"][entity["start"]:entity["end"]])
            dataset_part = doc["doc_id"].split("_")[1]
            key = dataset_part + "_" + entity["type"] + "_" + ne
            if "doc_id" not in items[key]:
                items[key]["doc_id"] = []
            if doc["doc_id"] not in items[key]["doc_id"]:
                items[key]["doc_id"].append(doc["doc_id"])
            if "dataset" not in items[key]:
                items[key]["dataset"] = []
            dataset_part = doc["doc_id"].split("_")[1]
            if dataset_part not in items[key]["dataset"]:
                items[key]["dataset"].append(dataset_part)
            items[key]["type"] = entity["type"]
            items[key]["content"] = ne
    return [
        Document(content=doc["content"],
                 meta={
                     "doc_id": doc["doc_id"],
                     "dataset": doc["dataset"],
                     "type": doc["type"],
                     "data_type": "gazetteers"
                 }) for doc in items.values()
    ]


def get_documents_from_sentences(docs):
    documents = []
    for doc in docs:
        dataset_part = doc["doc_id"].split("_")[1]
        documents.append(
            Document(content=" ".join(doc["tokens"]),
                     meta={
                         "entities": doc["entities"],
                         "data_type": "sentences",
                         "doc_id": [doc["doc_id"]],
                         "dataset": [dataset_part],
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


def prep_data(path, tokenizer, config: dict):
    dir_path = os.path.join(
        path, config["search_data_type"], config["search_algorithm"],
        "_".join([
            str(config[k]) for k in [
                "search_topk", "prepend_search_results", "filter_exact_match",
                "filter_same_document", "use_labels", "use_mentions"
            ]
        ]))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.exists(os.path.join(dir_path, "train.t5-small.jsonlines")):
        return os.path.join(dir_path,
                            "train.t5-small.jsonlines"), os.path.join(
                                dir_path,
                                "dev.t5-small.jsonlines"), os.path.join(
                                    thesis_path, "data/wnut/wnut_types.json")

    dataset_files = {
        "train": os.path.join(thesis_path, "data/wnut/wnut_train.json"),
        "dev": os.path.join(thesis_path, "data/wnut/wnut_dev.json"),
    }

    files = {"types": os.path.join(thesis_path, "data/wnut/wnut_types.json")}
    for key, file_path in dataset_files.items():
        files[key] = os.path.join(dir_path, os.path.basename(file_path))
        dir_name = os.path.dirname(files[key])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        shutil.copyfile(file_path, files[key])

    doc_store, search = setup_database(
        "_".join([
            config["search_data_type"], config["search_algorithm"],
            str(config["search_topk"])
        ]), config["search_algorithm"], config["search_topk"])

    documents = []
    # prepare database based on data type
    docs = []
    for part in ["train", "dev"]:
        with open(files[part], "r", encoding="utf-8") as file:
            d = json.load(file)
            if part == "dev":
                mask = np.zeros(len(d), dtype=int)
                mask[:len(d) // 2] = 1
                np.random.shuffle(mask)
                docs.extend(
                    [doc for m, doc in zip(mask.tolist(), d) if m == 0])
            else:
                docs.extend(d)
    if config["search_data_type"] == "gazetteers":
        documents = get_documents_from_gazetteers(docs)
    elif config["search_data_type"] == "sentences":
        documents = get_documents_from_sentences(docs)
    doc_store.write_documents(documents)

    if config["search_algorithm"] != "bm25":
        doc_store.update_embeddings(
            search.get_node("ANNRetriever"),  # type: ignore
            update_existing_embeddings=False)

    filters = {"train": ["train"], "dev": ["train", "dev"]}

    augment_dataset(tokenizer, files, filters, search, config["use_labels"],
                    config["use_mentions"], config["prepend_search_results"],
                    config["filter_exact_match"],
                    config["filter_same_document"])

    return files["tokenized_train"], files["tokenized_dev"], files["types"]


def run_tune_training(config: dict, fixed_params: dict):

    config.update(fixed_params)

    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    pl_seed.seed_everything(config["seed"])

    grad_accum_steps = factors(config["batch_size"])

    tokenizer = get_tokenizer(config)

    tokenized_train_data_file, tokenized_dev_data_file, type_data_file = prep_data(
        config["data_path"], tokenizer, config)

    processor = NERDataProcessor(config,
                                 tokenizer,
                                 tokenized_train_data_file,
                                 tokenized_dev_data_file,
                                 None,
                                 type_data_file,
                                 use_cache=False)
    config["num_labels"] = len(processor.labels)

    train, val, _ = processor.get_tensor_samples()
    config["train_len"] = len(train)

    collator = NERCollator(config["train_search_dropout"],
                           config["train_search_shuffle"])

    # Callbacks
    tune_report_f1 = TuneReportCallback(
        {
            # "train_f1": "train_f1",
            "val_f1": "val_f1_epoch",
        },
        on=["validation_end"])

    config["fused"] = True
    config["precision"] = "bf16-mixed"
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="", version=".")

    train_config = copy.deepcopy(config)
    trained = False
    while not trained:
        try:
            # Train loader

            train_loader = DataLoader(train,
                                      batch_size=train_config["batch_size"],
                                      collate_fn=collator,
                                      num_workers=3,
                                      persistent_workers=False,
                                      pin_memory=True,
                                      shuffle=True,
                                      prefetch_factor=20)
            # Validation loaders
            val_loader = DataLoader(
                val,
                batch_size=int(train_config["batch_size"] *
                               3) if train_config["batch_size"] > 1 else 8,
                collate_fn=ner_collate_fn,
                num_workers=3,
                persistent_workers=False,
                pin_memory=True,
                shuffle=False,
                prefetch_factor=20)

            trainer = pl.Trainer(
                accelerator="gpu",
                logger=tb_logger,
                devices=1,
                log_every_n_steps=train_config["batch_size"] *
                train_config["gradient_accumulation_steps"],
                accumulate_grad_batches=train_config[
                    "gradient_accumulation_steps"],
                precision=train_config["precision"],
                max_epochs=train_config["num_epochs"],
                check_val_every_n_epoch=4,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[tune_report_f1]  # type: ignore
            )

            model = ASPT5Model(train_config, tokenizer)

            trainer.fit(model, train_loader, val_dataloaders=val_loader)
            trained = True
        except Exception:
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]