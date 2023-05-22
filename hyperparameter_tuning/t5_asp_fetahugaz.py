import copy
import json
import sys
import os

from tqdm import tqdm

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
from hyperparameter_tuning.training import factors
from hyperparameter_tuning.ray_logging import TuneReportCallback
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl

from haystack import Pipeline, Document
from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore, BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, BM25Retriever

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768  #  384


def t5_asp_fetahugaz_configs():
    config = T5_BASE.copy()

    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning",
                                       "tune")
    config["name"] = "t5_asp_fetahugaz"
    config["batch_size"] = 40

    best_configs = [{
        "adam_weight_decay": 0.16654221006912023,
        "asp_dropout_rate": 0.255,
        "asp_hidden_dim": 764,
        "plm_learning_rate": 0.0017761923667238327,
        "search_algorithm": "ann",
        "search_topk": 23,
        "task_learning_rate": 0.002245444580169059,
        "train_search_dropout": 0.34795288270300007,
        "warmup_ratio": 0.40508103379041166,
        "num_epochs": 10
    }]

    config["asp_hidden_dim"] = tune.randint(100, 1000)
    config["asp_dropout_rate"] = tune.uniform(0.01, 0.5)
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["use_labels"] = True
    config["use_mentions"] = False
    config["prepend_search_results"] = False
    config["filter_exact_match"] = False
    config["filter_same_document"] = False
    config["search_data_type"] = "gazetteers"
    config["search_algorithm"] = tune.choice(["bm25", "ann"])
    config["search_topk"] = tune.randint(1, 40)
    config["seed"] = 42
    config["train_search_dropout"] = tune.uniform(0.0, 0.65)
    config["train_search_shuffle"] = False
    config["plm_learning_rate"] = tune.uniform(5e-6, 5e-3)
    config["task_learning_rate"] = tune.uniform(1e-5, 5e-3)
    config["adam_weight_decay"] = tune.uniform(5e-4, 0.17)
    config["warmup_ratio"] = tune.uniform(0.01, 0.5)
    config["num_epochs"] = tune.randint(10, 40)

    return config, best_configs


def create_faiss_document_store():
    document_store = FAISSDocumentStore(
        sql_url="postgresql://postgres:thesis123.@localhost:5432/fetahugaz",
        index="fetahugaz",
        faiss_index_factory_str="OPQ128_384,IVF20000,PQ128",
        embedding_dim=768,
        similarity="cosine")
    return document_store


def add_fetahugaz_documents(doc_store: BaseDocumentStore):
    if doc_store.get_document_count() == 0:
        documents = []
        with open(os.path.join(thesis_path, "data", "mlowner",
                               "lowner_gazetteer.json"),
                  "r",
                  encoding="utf-8") as file:
            fetahu_gaz = json.load(file)
        for idx, gaz in tqdm(enumerate(fetahu_gaz), total=len(fetahu_gaz)):
            documents.append(
                Document(content=gaz["entity"],
                         meta={
                             "data_type": "gazetteers",
                             "type": gaz["type"],
                             "entity_id": gaz["entity_id"]
                         }))
            if idx > 0 and idx % 1e4 == 0:
                doc_store.write_documents(documents)
                documents = []
        doc_store.write_documents(documents)


def train_update_faiss_index(document_store: FAISSDocumentStore,
                             retriever: EmbeddingRetriever):
    documents = document_store.get_all_documents()
    embeddings = retriever.embed_documents(documents)
    document_store.train_index(embeddings=embeddings)
    document_store.update_embeddings(retriever)
    document_store.save(index_path=os.path.join(thesis_path, "search",
                                                "fetahugaz",
                                                "faiss_index.faiss"),
                        config_path=os.path.join(thesis_path, "search",
                                                 "fetahugaz",
                                                 "faiss_config.json"))


def setup_database(search_algorithm: str, search_topk: int):
    search = Pipeline()
    if search_algorithm == "bm25":
        document_store = ElasticsearchDocumentStore(
            index="fetahugaz", embedding_dim=EMBEDDING_DIM)
        bm25_retriever = BM25Retriever(document_store, top_k=search_topk)
        search.add_node(component=bm25_retriever,
                        name="BM25Retriever",
                        inputs=["Query"])
    elif search_algorithm.startswith("ann"):
        document_store = FAISSDocumentStore.load(
            index_path=os.path.join(thesis_path, "search", "fetahugaz",
                                    "faiss_index.faiss"),
            config_path=os.path.join(thesis_path, "search", "fetahugaz",
                                     "faiss_config.json"))
        ann_retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format="sentence_transformers",
            top_k=search_topk)
        search.add_node(component=ann_retriever,
                        name="ANNRetriever",
                        inputs=["Query"])

    if len(search.components) == 0:
        raise Exception(
            "Argument error: search_algorithm - must be: bm25 | ann , but is: "
            + search_algorithm)

    return search


def augment_dataset(
    data_path,
    tokenizer,
    files,
    parts,
    database,
    use_labels,
    use_mentions,
    prepend_search_results,
):
    for part in parts:
        tokenized_name = "tokenized_" + part
        files[tokenized_name] = tokenize_database_json(
            tokenizer,
            files[part],
            files["types"],
            database,
            False,
            False,
            use_labels,
            use_mentions,
            data_path,
            prepend_search_results=prepend_search_results)


def prep_data(path, tokenizer, config: dict):
    data_path = os.path.join(
        path, config["name"], config["search_algorithm"], "_".join([
            str(config[k]) for k in [
                "search_topk", "prepend_search_results", "filter_exact_match",
                "filter_same_document", "use_labels", "use_mentions"
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

    search = setup_database(config["search_algorithm"], config["search_topk"])

    parts = ["train", "dev"]

    augment_dataset(data_path, tokenizer, files, parts, search,
                    config["use_labels"], config["use_mentions"],
                    config["prepend_search_results"])

    del search

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return files["tokenized_train"], files["tokenized_dev"], os.path.join(
        thesis_path, "data", "mlowner", "lowner_types.json")


def run_t5_asp_fetahugaz_training(config: dict, fixed_params: dict):
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
            trained = True
        except Exception:
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]
