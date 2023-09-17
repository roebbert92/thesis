import json
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
sys.path.append(thesis_path)

import shutil
import torch
from torch.utils.data import DataLoader, Dataset
import json
from haystack import Pipeline
from haystack.nodes import JoinDocuments
import pickle
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data_preprocessing.tensorize import NERDataProcessor, ner_collate_fn
from data_preprocessing.tokenize import tokenize_search_results_json
from models.asp_t5 import ASPT5Model, get_tokenizer
from configs.asp_t5 import BEST_WNUT_T5_ASP_LOWNERGAZ_SENT
from hyperparameter_tuning.utils import get_search_results
from search.lownergaz.setup import add_lownergaz_search_components
from search.sent.setup import add_sent_search_components

eval_gazetteer_contents = {
    2: ("lownergaz_sent", "wnut_train", "wnut_dev"),
    3: ("lownergaz_sent", "wnut_train", "wnut_dev", "wnut_test")
}

content_added = ["gaz", "sent", "both"]

train_gazetteer_contents = [
    ("lownergaz_sent", ),
    ("wnut_train", ),
    ("lownergaz_sent", "wnut_train"),
]

seeds = [1, 2, 3]

files = {
    "types": os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
    "wnut_test": os.path.join(thesis_path, "data", "wnut", "wnut_test.json"),
}

with open(files["wnut_test"], encoding="utf-8") as file:
    wnut_test = json.load(file)

datasets = {"wnut_test": wnut_test}

gazetteer, finetuning, pretrained, config = (True, "full", True,
                                             BEST_WNUT_T5_ASP_LOWNERGAZ_SENT)
config.update({
    "data_path":
    os.path.join(thesis_path, "experiments", "03_adaptation_emerging_entities",
                 "data")
})


def get_validation_dataloader(config, dataset: Dataset):
    return DataLoader(dataset,
                      batch_size=int(config["batch_size"] * 2),
                      collate_fn=ner_collate_fn,
                      num_workers=3,
                      persistent_workers=False,
                      pin_memory=True,
                      shuffle=False,
                      prefetch_factor=20)


def test_model(config, best_ckpt_path, last_ckpt_path, dataset: Dataset, name):
    metrics_base_path = os.path.join(
        config["data_path"], f"seed_{str(seed)}",
        "06_cross_gazetteer_content_eval",
        f"{train_gaz_idx}_{timestep}_{content_location}")
    os.makedirs(metrics_base_path, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(thesis_path, "experiments",
                              "03_adaptation_emerging_entities",
                              "lightning_logs"),
        name="_".join([
            str(seed),
            f"content_eval_{gazetteer}_{finetuning}_{pretrained}_{config['name']}",
            str(train_gaz_idx)
        ]),
        version=f"{timestep}_{content_location}_" + name)
    trainer = pl.Trainer(accelerator="gpu",
                         logger=tb_logger,
                         devices=1,
                         precision="bf16-mixed",
                         num_sanity_val_steps=0,
                         enable_checkpointing=False,
                         enable_progress_bar=True)
    val_loader = get_validation_dataloader(config, dataset)

    def save_metrics(model, dataset, checkpoint):
        with open(
                os.path.join(metrics_base_path, f"{checkpoint}_{dataset}.pkl"),
                "wb") as file:
            pickle.dump(model.test_metrics, file)

    last_ckpt_epoch = torch.load(last_ckpt_path)["epoch"] - 1
    best_ckpt_epoch = torch.load(best_ckpt_path)["epoch"]

    # test model
    last_model = ASPT5Model.load_from_checkpoint(last_ckpt_path)
    trainer.test(last_model, val_loader)
    save_metrics(last_model, name, "last")
    if last_ckpt_epoch != best_ckpt_epoch:
        best_model = ASPT5Model.load_from_checkpoint(best_ckpt_path)
        trainer.test(best_model, val_loader)
        save_metrics(best_model, name, "best")
    else:
        shutil.copy(os.path.join(metrics_base_path, f"last_{name}.pkl"),
                    os.path.join(metrics_base_path, f"best_{name}.pkl"))


def setup_database(sent_search_algorithm: str, sent_search_topk: int,
                   gaz_search_algorithm: str, gaz_search_topk: int,
                   join_method: str, join_topk: int, content_location: str):
    search = Pipeline()
    join_documents_input = []
    # sentences
    sent_name = "sent" if content_location == "gaz" else eval_gaz_name + "_sent"
    add_sent_search_components(search,
                               sent_search_algorithm,
                               sent_search_topk,
                               join_documents_input,
                               name=sent_name)

    # lowner gazetteers
    lownergaz_name = "lownergaz" if content_location == "sent" else eval_gaz_name + "_lownergaz"
    add_lownergaz_search_components(search,
                                    gaz_search_algorithm,
                                    gaz_search_topk,
                                    join_documents_input,
                                    name=lownergaz_name)

    # join documents

    join_documents = JoinDocuments(join_mode=join_method, top_k_join=join_topk)
    search.add_node(join_documents, "DocumentJoin", join_documents_input)

    return search


def get_tokenized_filepath(config, file_path, search_results, data_path):
    tokenizer = get_tokenizer(config)
    use_labels = config["use_labels"] if "use_labels" in config else None
    use_mentions = config["use_mentions"] if "use_mentions" in config else None
    sent_use_labels = config[
        "sent_use_labels"] if "sent_use_labels" in config else None
    sent_use_mentions = config[
        "sent_use_mentions"] if "sent_use_mentions" in config else None
    gaz_use_labels = config[
        "gaz_use_labels"] if "gaz_use_labels" in config else None
    gaz_use_mentions = config[
        "gaz_use_mentions"] if "gaz_use_mentions" in config else None

    return tokenize_search_results_json(tokenizer,
                                        file_path,
                                        files["types"],
                                        search_results,
                                        data_path,
                                        use_labels=use_labels,
                                        use_mentions=use_mentions,
                                        sent_use_labels=sent_use_labels,
                                        sent_use_mentions=sent_use_mentions,
                                        gaz_use_labels=gaz_use_labels,
                                        gaz_use_mentions=gaz_use_mentions,
                                        prepend_search_results=False)


for timestep, eval_gaz_comb in eval_gazetteer_contents.items():
    eval_gaz_name = "_".join(eval_gaz_comb)
    for content_location in content_added:
        # get search for content location
        search_base_path = os.path.join(config["data_path"],
                                        "01_search_results",
                                        "06_content_location",
                                        f"{content_location}_{timestep}")
        os.makedirs(search_base_path, exist_ok=True)
        # query search results
        test_search_path = os.path.join(search_base_path, "wnut_test.pkl")
        if not os.path.exists(test_search_path):
            search = setup_database(config["sent_search_algorithm"],
                                    config["sent_search_topk"],
                                    config["gaz_search_algorithm"],
                                    config["gaz_search_topk"],
                                    config["search_join_method"],
                                    config["search_topk"],
                                    content_location=content_location)
            search_results_test = get_search_results(search,
                                                     datasets["wnut_test"])
            with open(test_search_path, "wb") as file:
                pickle.dump(search_results_test, file)
        else:
            with open(test_search_path, "rb") as file:
                search_results_test = pickle.load(file)
        # tokenize
        tokenized_data_path = os.path.join(config["data_path"],
                                           "02_tokenized_dataset",
                                           "06_cross_gazetteer_content_eval",
                                           f"{content_location}_{timestep}")
        os.makedirs(tokenized_data_path, exist_ok=True)
        wnut_test_tokenized_path = os.path.join(
            config["data_path"], "02_tokenized_dataset",
            "06_cross_gazetteer_content_eval",
            f"{content_location}_{timestep}", "wnut_test.t5-small.jsonlines")
        get_tokenized_filepath(config, files["wnut_test"], search_results_test,
                               tokenized_data_path)

        # get dataset
        processor = NERDataProcessor(
            config,
            get_tokenizer(config),
            os.path.join(config["data_path"], "02_tokenized_dataset",
                         "ann_6_bm25_12_reciprocal_rank_fusion_16", "0_0",
                         "wnut_train.t5-small.jsonlines"),
            os.path.join(config["data_path"], "02_tokenized_dataset",
                         "ann_6_bm25_12_reciprocal_rank_fusion_16", "0_0",
                         "wnut_dev.t5-small.jsonlines"),
            wnut_test_tokenized_path,
            files["types"],
            use_cache=False)
        _, _, test_dataset = processor.get_tensor_samples()
        for seed in seeds:
            for train_gaz_idx, train_gaz in enumerate(
                    train_gazetteer_contents):
                checkpoint_base_path = os.path.join(
                    config["data_path"], f"seed_{str(seed)}", "03_checkpoints",
                    "True_full_True_t5_asp_lownergaz_sent", f"{train_gaz_idx}")
                best_ckpt_path = os.path.join(checkpoint_base_path,
                                              "best.ckpt")
                last_ckpt_path = os.path.join(checkpoint_base_path,
                                              "last.ckpt")

                test_model(config, best_ckpt_path, last_ckpt_path,
                           test_dataset, "wnut_test")
