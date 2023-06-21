import copy
import pickle
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from configs.asp_t5 import T5_BASE
from models.asp_t5 import ASPT5Model, get_tokenizer

from ray import tune
import torch
from torch.utils.data import DataLoader
from data_preprocessing.tensorize import NERCollator, NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_search_results_json
from hyperparameter_tuning.utils import factors, get_search_results, get_search_results_for_file
from hyperparameter_tuning.t5_asp_lownergaz import setup_database as gaz_setup_database
from hyperparameter_tuning.t5_asp_sent import setup_database as sent_setup_database
from hyperparameter_tuning.ray_logging import TuneReportCallback
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from search.lownergaz.setup import add_lownergaz_search_components
from search.sent.setup import add_sent_search_components
from haystack import Pipeline
from haystack.nodes import JoinDocuments


def best_pretrained_wnut_t5_asp_configs(pretrained_ckpt_path: str):
    config = T5_BASE.copy()

    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning",
                                       "tune")
    config["name"] = "best_wnut_t5_asp_lownergaz_sent"
    config["batch_size"] = 40

    best_configs = [{
        "adam_weight_decay": 0.011738749999999989,
        "asp_dropout_rate": 0.4540625,
        "asp_hidden_dim": 633,
        "gaz_search_algorithm": "bm25",
        "gaz_search_topk": 4,
        "gaz_use_mentions": True,
        "num_epochs": 16,
        "plm_learning_rate": 0.00017496219281663535,
        "search_join_method": "reciprocal_rank_fusion",
        "search_topk": 8,
        "sent_search_algorithm": "ann",
        "sent_search_topk": 4,
        "sent_use_mentions": True,
        "task_learning_rate": 0.0035849253731343286,
        "train_search_dropout": 0.05492957746478871,
        "warmup_ratio": 0.37917808219178084
    }]

    config["asp_hidden_dim"] = 633
    config["asp_dropout_rate"] = tune.uniform(0.2, 0.5)
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["sent_search_algorithm"] = "ann"
    config["sent_search_topk"] = tune.randint(3, 7)
    config["sent_use_labels"] = True
    config["sent_use_mentions"] = True
    config["gaz_search_algorithm"] = "bm25"
    config["gaz_search_topk"] = tune.randint(4, 13)
    config["gaz_use_labels"] = True
    config["gaz_use_mentions"] = True
    config["search_join_method"] = "reciprocal_rank_fusion"
    config["search_topk"] = 20
    config["prepend_search_results"] = False
    config["filter_exact_match"] = False
    config["filter_same_document"] = False
    config["seed"] = 42
    config["train_search_dropout"] = tune.uniform(0.01, 0.4)
    config["train_search_shuffle"] = False
    config["plm_learning_rate"] = tune.uniform(5e-6, 5e-4)
    config["task_learning_rate"] = tune.uniform(1e-4, 1e-2)
    config["adam_weight_decay"] = tune.uniform(5e-5, 0.02)
    config["warmup_ratio"] = tune.uniform(0.01, 0.4)
    config["num_epochs"] = 15

    config["ckpt_path"] = pretrained_ckpt_path

    return config, best_configs


def worst_pretrained_wnut_t5_asp_configs(pretrained_ckpt_path: str):
    config = T5_BASE.copy()

    config["data_path"] = os.path.join(thesis_path, "hyperparameter_tuning",
                                       "tune")
    config["name"] = "worst_wnut_t5_asp_lownergaz_sent"
    config["batch_size"] = 40

    best_configs = [{
        "adam_weight_decay": 0.011738749999999989,
        "asp_dropout_rate": 0.4540625,
        "asp_hidden_dim": 633,
        "gaz_search_algorithm": "bm25",
        "gaz_search_topk": 4,
        "gaz_use_mentions": True,
        "num_epochs": 16,
        "plm_learning_rate": 0.00017496219281663535,
        "search_join_method": "reciprocal_rank_fusion",
        "search_topk": 8,
        "sent_search_algorithm": "ann",
        "sent_search_topk": 4,
        "sent_use_mentions": True,
        "task_learning_rate": 0.0035849253731343286,
        "train_search_dropout": 0.05492957746478871,
        "warmup_ratio": 0.37917808219178084
    }]

    config["asp_hidden_dim"] = 633
    config["asp_dropout_rate"] = tune.uniform(0.2, 0.5)
    config["asp_init_std"] = 0.02
    config["asp_activation"] = "relu"
    config["beam_size"] = 1
    config["sent_search_algorithm"] = "ann"
    config["sent_search_topk"] = tune.randint(3, 7)
    config["sent_use_labels"] = True
    config["sent_use_mentions"] = True
    config["gaz_search_algorithm"] = "bm25"
    config["gaz_search_topk"] = tune.randint(4, 13)
    config["gaz_use_labels"] = True
    config["gaz_use_mentions"] = True
    config["search_join_method"] = "reciprocal_rank_fusion"
    config["search_topk"] = 20
    config["prepend_search_results"] = False
    config["filter_exact_match"] = False
    config["filter_same_document"] = False
    config["seed"] = 42
    config["train_search_dropout"] = tune.uniform(0.01, 0.4)
    config["train_search_shuffle"] = False
    config["plm_learning_rate"] = tune.uniform(5e-6, 5e-4)
    config["task_learning_rate"] = tune.uniform(1e-4, 1e-2)
    config["adam_weight_decay"] = tune.uniform(5e-5, 0.02)
    config["warmup_ratio"] = tune.uniform(0.01, 0.4)
    config["num_epochs"] = tune.randint(5, 21)

    config["ckpt_path"] = pretrained_ckpt_path

    return config, best_configs


def setup_database(sent_search_algorithm: str,
                   sent_search_topk: int,
                   gaz_search_algorithm: str,
                   gaz_search_topk: int,
                   join_method: str,
                   join_topk: int,
                   reset=False,
                   name: str = "lownergaz"):
    search = Pipeline()
    join_documents_input = []
    # sentences
    sent_name = "sent" if name == "lownergaz" else name + "_sent"
    add_sent_search_components(search, sent_search_algorithm, sent_search_topk,
                               join_documents_input, reset, sent_name)

    # lowner gazetteers
    lownergaz_name = name if name == "lownergaz" else name + "_lownergaz"
    add_lownergaz_search_components(search, gaz_search_algorithm,
                                    gaz_search_topk, join_documents_input,
                                    reset, lownergaz_name)

    # join documents

    join_documents = JoinDocuments(join_mode=join_method, top_k_join=join_topk)
    search.add_node(join_documents, "DocumentJoin", join_documents_input)

    return search


def augment_dataset(config, data_path, tokenizer, files, parts):
    join_documents = JoinDocuments(join_mode=config["search_join_method"],
                                   top_k_join=config["search_topk"])
    for part in parts:
        # load lowner search results
        wnut_gaz_result_path = os.path.join(
            thesis_path, "search", "lownergaz",
            f"wnut_{part}_{config['gaz_search_algorithm']}.pkl")
        if not os.path.exists(wnut_gaz_result_path):
            search = gaz_setup_database(config["gaz_search_algorithm"], 50)
            wnut_result = get_search_results_for_file(search, files[part])
            with open(wnut_gaz_result_path, "wb") as file:
                pickle.dump(wnut_result, file)
            del search
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            with open(wnut_gaz_result_path, "rb") as file:
                wnut_result: dict = pickle.load(file)
        # process search results - top k
        wnut_results = {
            key: value[:config["gaz_search_topk"]]
            for key, value in wnut_result.items()
        }
        # load lowner search results
        wnut_sent_result_path = os.path.join(
            thesis_path, "search", "sent",
            f"wnut_{part}_{config['sent_search_algorithm']}.pkl")
        if not os.path.exists(wnut_sent_result_path):
            search = sent_setup_database(config["sent_search_algorithm"], 50)
            sent_result = get_search_results_for_file(search, files[part])
            with open(wnut_sent_result_path, "wb") as file:
                pickle.dump(sent_result, file)
            del search
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            with open(wnut_sent_result_path, "rb") as file:
                sent_result: dict = pickle.load(file)
        # process search results - top k
        sent_results = {
            key: value[:config["sent_search_topk"]]
            for key, value in sent_result.items()
        }

        search_results = {
            key: join_documents.run([{
                "documents": wnut_results[key]
            }, {
                "documents": sent_results[key]
            }])[0]["documents"]
            for key in wnut_results
        }

        tokenized_name = "tokenized_" + part
        files[tokenized_name] = tokenize_search_results_json(
            tokenizer,
            files[part],
            files["types"],
            search_results,
            data_path,
            sent_use_labels=config["sent_use_labels"],
            sent_use_mentions=config["sent_use_mentions"],
            gaz_use_labels=config["gaz_use_labels"],
            gaz_use_mentions=config["gaz_use_mentions"],
            prepend_search_results=config["prepend_search_results"])


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
            os.path.join(data_path, "wnut_train.t5-small.jsonlines")):
        return os.path.join(data_path,
                            "wnut_train.t5-small.jsonlines"), os.path.join(
                                data_path,
                                "wnut_dev.t5-small.jsonlines"), os.path.join(
                                    thesis_path, "data", "wnut",
                                    "wnut_types.json")

    files = {
        "types": os.path.join(thesis_path, "data", "wnut", "wnut_types.json"),
        "train": os.path.join(thesis_path, "data", "wnut", "wnut_train.json"),
        "dev": os.path.join(thesis_path, "data", "wnut", "wnut_dev.json"),
    }

    parts = ["train", "dev"]

    augment_dataset(config, data_path, tokenizer, files, parts)

    return files["tokenized_train"], files["tokenized_dev"], files["types"]


def run_pretrained_wnut_t5_asp_lownergaz_sent_training(config: dict,
                                                       fixed_params: dict):
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
    tune_report_f1 = TuneReportCallback({"f1": "val_f1"},
                                        on=["validation_end"])

    config["fused"] = True
    config["precision"] = "bf16-mixed"
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

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

            model = ASPT5Model.load_from_checkpoint(config["ckpt_path"],
                                                    config=train_config,
                                                    tokenizer=tokenizer)

            trainer.fit(model, train_loader, val_dataloaders=val_loader)
            trainer.validate(model, val_loader)
            trained = True
        except Exception:
            train_config["gradient_accumulation_steps"] = grad_accum_steps[
                grad_accum_steps.index(
                    train_config["gradient_accumulation_steps"]) + 1]
            train_config["batch_size"] = train_config[
                "batch_size"] // train_config["gradient_accumulation_steps"]
