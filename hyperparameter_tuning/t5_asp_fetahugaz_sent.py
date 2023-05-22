import copy
import json
import sys
import os

from tqdm import tqdm
from hyperparameter_tuning.t5_asp_gaz import add_multiconer_gazetteers

from hyperparameter_tuning.t5_asp_sent import add_multiconer_sentences

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from configs.asp_t5 import T5_BASE
from models.asp_t5 import ASPT5Model, get_tokenizer

from ray import tune
import torch
from torch.utils.data import DataLoader
from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_database_json, tokenize_json
from hyperparameter_tuning.training import factors, get_gazetteers_from_documents, get_sentences_from_documents
from hyperparameter_tuning.ray_logging import TuneReportCallback
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl

from haystack import Pipeline, Document
from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore, BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, BM25Retriever, JoinDocuments
import faiss

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def t5_asp_fetahugaz_sent_configs():
    config = T5_BASE.copy()

    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning",
                                       "tune")
    config["name"] = "t5_asp_fetahugaz_sent"
    config["batch_size"] = 40

    best_configs = [{
        'asp_hidden_dim': 142,
        'task_learning_rate': 0.0013480523331922776,
        'adam_weight_decay': 0.49637507889057786,
        'warmup_ratio': 0.184451637360714,
        'sent_search_algorithm': 'ann',
        'sent_search_topk': 8,
        'sent_use_labels': True,
        'sent_use_mentions': True,
        'gaz_search_algorithm': 'ann',
        'gaz_search_topk': 8,
        'gaz_use_labels': True,
        'gaz_use_mentions': True,
        'search_join_method': 'concatenate',
        'search_topk': 6,
        'train_search_dropout': 0.21126587935893093,
        'asp_dropout_rate': 0.3,
        'plm_learning_rate': 5e-05,
        'num_epochs': 20
    }]

    config["asp_hidden_dim"] = tune.randint(100, 1000)
    config["asp_dropout_rate"] = tune.uniform(0.01, 0.5)
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["sent_search_algorithm"] = tune.choice(["bm25", "ann"])
    config["sent_search_topk"] = tune.randint(1, 10)
    config["sent_use_labels"] = tune.choice([True, False])
    config["sent_use_mentions"] = tune.choice([True, False])
    config["gaz_search_algorithm"] = tune.choice(["bm25", "ann"])
    config["gaz_search_topk"] = tune.randint(1, 15)
    config["gaz_use_labels"] = tune.choice([True, False])
    config["gaz_use_mentions"] = tune.choice([True, False])
    config["search_join_method"] = tune.choice(
        ["concatenate", "merge", "reciprocal_rank_fusion"])
    config["search_topk"] = tune.randint(5, 20)
    config["prepend_search_results"] = False
    config["filter_exact_match"] = False
    config["filter_same_document"] = False
    config["seed"] = 42
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["train_search_shuffle"] = False
    config["plm_learning_rate"] = tune.uniform(5e-6, 5e-3)
    config["task_learning_rate"] = tune.uniform(1e-5, 5e-3)
    config["adam_weight_decay"] = tune.uniform(5e-4, 0.5)
    config["warmup_ratio"] = tune.uniform(0.01, 0.5)
    config["num_epochs"] = tune.randint(10, 40)

    return config, best_configs


def setup_database(sent_search_algorithm: str, sent_search_topk: int,
                   gaz_search_algorithm: str, gaz_search_topk: int,
                   join_method: str, join_topk: int):
    search = Pipeline()
    join_documents_input = []
    # sentences
    if sent_search_algorithm == "bm25":
        sent_document_store = ElasticsearchDocumentStore(
            index="sent", embedding_dim=EMBEDDING_DIM, similarity="cosine")

        bm25_retriever = BM25Retriever(sent_document_store,
                                       top_k=sent_search_topk)
        search.add_node(component=bm25_retriever,
                        name="SentBM25Retriever",
                        inputs=["Query"])
        join_documents_input.append("SentBM25Retriever")
    elif sent_search_algorithm.startswith("ann"):
        document_store = FAISSDocumentStore.load(
            index_path=os.path.join(thesis_path, "search", "sent",
                                    "faiss_index.faiss"),
            config_path=os.path.join(thesis_path, "search", "sent",
                                     "faiss_config.json"))
        document_store.faiss_indexes[
            document_store.index] = faiss.index_cpu_to_all_gpus(
                index=document_store.faiss_indexes[document_store.index])
        ann_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=sent_search_topk)
        search.add_node(component=ann_retriever,
                        name="SentANNRetriever",
                        inputs=["Query"])
        join_documents_input.append("SentANNRetriever")

    # fetahu gazetters
    if gaz_search_algorithm == "bm25":
        document_store = ElasticsearchDocumentStore(
            index="fetahugaz", embedding_dim=EMBEDDING_DIM)
        bm25_retriever = BM25Retriever(document_store, top_k=gaz_search_topk)
        search.add_node(component=bm25_retriever,
                        name="GazBM25Retriever",
                        inputs=["Query"])
        join_documents_input.append("GazBM25Retriever")
    elif gaz_search_algorithm.startswith("ann"):
        document_store = FAISSDocumentStore.load(
            index_path=os.path.join(thesis_path, "search", "fetahugaz",
                                    "faiss_index.faiss"),
            config_path=os.path.join(thesis_path, "search", "fetahugaz",
                                     "faiss_config.json"))
        ann_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=gaz_search_topk)
        search.add_node(component=ann_retriever,
                        name="GazANNRetriever",
                        inputs=["Query"])
        join_documents_input.append("GazANNRetriever")

    # join documents

    join_documents = JoinDocuments(join_mode=join_method, top_k_join=join_topk)
    search.add_node(join_documents, "DocumentJoin", join_documents_input)

    if len(search.components) == 0:
        raise Exception(
            "Argument error: search_algorithm - must be: bm25 | ann , but is: "
            + sent_search_algorithm)

    return search


def augment_dataset(
    data_path,
    tokenizer,
    files,
    parts,
    database,
    sent_use_labels,
    sent_use_mentions,
    gaz_use_labels,
    gaz_use_mentions,
    prepend_search_results,
):
    for part in parts:
        tokenized_name = "tokenized_" + part
        files[tokenized_name] = tokenize_database_json(
            tokenizer,
            files[part],
            files["types"],
            database,
            sent_use_labels,
            sent_use_mentions,
            gaz_use_labels,
            gaz_use_mentions,
            data_path,
            prepend_search_results=prepend_search_results)


def prep_data(path, tokenizer, config: dict):
    data_path = os.path.join(
        path, config["name"],
        config["sent_search_algorithm"] + "_" + config["gaz_search_algorithm"],
        "_".join([
            str(int(config[k]) if isinstance(config[k], bool) else config[k])
            for k in [
                "search_join_method", "search_topk", "sent_search_topk",
                "gaz_search_topk", "prepend_search_results",
                "filter_exact_match", "filter_same_document",
                "sent_use_labels", "sent_use_mentions", "gaz_use_labels",
                "gaz_use_mentions"
            ]
        ]))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    elif os.path.exists(
            os.path.join(data_path, "lowner_train.t5-small.jsonlines")):
        return os.path.join(data_path,
                            "lowner_train.t5-small.jsonlines"), os.path.join(
                                data_path,
                                "lowner_dev.t5-small.jsonlines"), os.path.join(
                                    thesis_path, "data", "mlowner",
                                    "lowner_types.json")

    files = {
        "types": os.path.join(thesis_path, "data", "mlowner",
                              "lowner_types.json"),
        "train": os.path.join(thesis_path, "data", "mlowner",
                              "lowner_train.json"),
        "dev": os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
    }

    search = setup_database(config["sent_search_algorithm"],
                            config["sent_search_topk"],
                            config["gaz_search_algorithm"],
                            config["gaz_search_topk"],
                            config["search_join_method"],
                            config["search_topk"])

    parts = ["train", "dev"]

    augment_dataset(data_path, tokenizer, files, parts, search,
                    config["sent_use_labels"], config["sent_use_mentions"],
                    config["gaz_use_labels"], config["gaz_use_mentions"],
                    config["prepend_search_results"])

    del search

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return files["tokenized_train"], files["tokenized_dev"], os.path.join(
        thesis_path, "data", "mlowner", "lowner_types.json")


def run_t5_asp_fetahugaz_sent_training(config: dict, fixed_params: dict):
    config.update(fixed_params)

    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    seed_everything(config["seed"])

    grad_accum_steps = factors(config["batch_size"])

    tokenizer = get_tokenizer(config)

    [tokenized_train_data_file, tokenized_dev_data_file,
     type_data_file] = prep_data(config["data_path"], tokenizer, config)

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

    # Callbacks
    tune_report_f1 = TuneReportCallback({"val_f1": "val_f1"},
                                        on=["validation_end"])

    config["fused"] = True
    config["precision"] = "bf16-mixed"
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="", version=".")

    collator = NERCollator(config["train_search_dropout"],
                           config["train_search_shuffle"])

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
            val_loader = DataLoader(val,
                                    batch_size=int(train_config["batch_size"] *
                                                   4),
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
                check_val_every_n_epoch=2,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[tune_report_f1]  # type: ignore
            )

            model = ASPT5Model(train_config, tokenizer)

            trainer.fit(model, train_loader, val_dataloaders=val_loader)
            trainer.validate(model, val_loader)
            trained = True
        except Exception:
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]
