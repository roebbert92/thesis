import pickle
import sys
import os

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from collections import defaultdict
from typing import Any, List, Optional
import lightning.pytorch as pl
from nltk import ngrams
from models.metrics import ASPMetrics
import random
import json
import shutil
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger


class DictMatchDataset(Dataset):

    def __init__(self, json_path: str) -> None:
        super().__init__()
        with open(json_path, "r") as file:
            self.items = json.load(file)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> Any:
        return self.items[index]


def collate_dict_batch(batch: List[dict]):
    return [item["doc_id"]
            for item in batch], [item["tokens"] for item in batch
                                 ], [[(ent["start"], ent["end"], ent["type"])
                                      for ent in item["entities"]]
                                     for item in batch]


class DictMatch(pl.LightningModule):

    def __init__(self, gazetteer_paths: List[str], seed: int) -> None:
        super().__init__()
        gazetteer = []
        for gazetteer_path in gazetteer_paths:
            with open(gazetteer_path, "r") as file:
                gazetteer.extend(json.load(file))
        gaz_init = defaultdict(set)
        for gaz in gazetteer:
            if "entity" in gaz:
                gaz_init[" ".join(gaz["entity"].strip().split(" ")).lower()].add(
                    gaz["type"])
            elif "entities" in gaz:
                for ent in gaz["entities"]:
                    gaz_init[" ".join(
                        gaz["tokens"][ent["start"]:ent["end"]]).lower()].add(
                            ent["type"])
            elif "content" in gaz:
                for ent in gaz["meta"]["entities"]:
                    gaz_init[" ".join(gaz["content"].strip().split(
                        " ")[ent["start"]:ent["end"]]).lower()].add(ent["type"])
            self.gaz = dict(gaz_init)
        self.test_metrics = ASPMetrics()
        self.seed = seed

    def forward(self, batch):
        batch_predictions = []
        for item in batch:
            predictions = []
            not_predicted = [[(idx, t) for idx, t in enumerate(item)]]
            for n_gram_len in range(len(item), 0, -1):
                n_grams = []
                for available in not_predicted:
                    if len(available) >= n_gram_len:
                        n_grams.extend(ngrams(available, n_gram_len))
                for n_gram in n_grams:
                    start = n_gram[0][0]
                    end = n_gram[-1][0]
                    n_gram_tokens = " ".join(t[1] for t in n_gram).lower()
                    if n_gram_tokens in self.gaz and not (start, end) in [
                        (pred[0], pred[1]) for pred in predictions
                    ]:
                        predictions.append(
                            (start, end + 1,
                             random.choice(list(self.gaz[n_gram_tokens]))))
                not_predicted = [[]]
                current_prediction_idx = -1
                start_predictions = [pred[0] for pred in predictions]
                idx = 0
                while idx < len(item):
                    try:
                        current_prediction_idx = start_predictions.index(idx)
                    except ValueError:
                        current_prediction_idx = -1
                    if current_prediction_idx == -1:
                        not_predicted[-1].append((idx, item[idx]))
                        idx += 1
                    else:
                        if len(not_predicted[-1]) > 0:
                            not_predicted.append([])
                        idx += predictions[current_prediction_idx][
                            1] - predictions[current_prediction_idx][0] + 1
                        current_prediction_idx = -1
            batch_predictions.append(predictions)
        return batch_predictions

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_metrics.reset()
        if "PL_GLOBAL_SEED" in os.environ:
            del os.environ["PL_GLOBAL_SEED"]
        pl.seed_everything(self.seed)

    def test_step(self, batch, batch_idx):
        doc_ids, tokens, labels = batch
        predictions = self.forward(tokens)
        for doc_id, preds, targets in zip(doc_ids, predictions, labels):
            self.test_metrics.update(doc_id, preds, targets)

    def on_test_epoch_end(self) -> None:
        errors = self.test_metrics.errors()
        f1 = self.test_metrics.f1()
        precision = self.test_metrics.precision()
        recall = self.test_metrics.recall()
        self.log_dict(
            {
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "test_error_type1": errors[0],
                "test_error_type2": errors[1],
                "test_error_type3": errors[2],
                "test_error_type4": errors[3],
                "test_error_type5": errors[4],
            },
            logger=True,
            on_epoch=True)
        super().on_test_epoch_end()

    def get_dataloader(self, json_path: str, batch_size: int):
        dataset = DictMatchDataset(json_path)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=3,
                                shuffle=False,
                                collate_fn=collate_dict_batch)

        return dataloader


def experiment01(gazetteer_name: str, gazetteer_paths: List[str]):
    seeds = [1, 2, 3]
    config = {
        "name":
        f"dict_match_{gazetteer_name}",
        "gazetteer_paths": [*gazetteer_paths],
        "batch_size":
        40,
        "data_path":
        os.path.join(thesis_path, "experiments", "01_performance", "data"),
    }

    files = {
        "types":
        os.path.join(thesis_path, "data", "mlowner", "lowner_types.json"),
        "train":
        os.path.join(thesis_path, "data", "mlowner", "lowner_train.json"),
        "dev":
        os.path.join(thesis_path, "data", "mlowner", "lowner_dev.json"),
        "test":
        os.path.join(thesis_path, "data", "mlowner", "lowner_test.json"
                     #"lowner_dev.json"
                     ),
    }

    for seed in seeds:
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(thesis_path, "experiments", "01_performance",
                                  "lightning_logs"),
            name="_".join([str(seed), config["name"]]),
        )
        model = DictMatch(config["gazetteer_paths"], seed)
        trainer = pl.Trainer(accelerator="cpu", logger=tb_logger)
        metrics_base_path = os.path.join(config["data_path"],
                                         f"seed_{str(seed)}", "04_metrics",
                                         config["name"])
        os.makedirs(metrics_base_path, exist_ok=True)

        def save_metrics(dataset):
            with open(os.path.join(metrics_base_path, f"last_{dataset}.pkl"),
                      "wb") as file:
                pickle.dump(model.test_metrics, file)
            shutil.copy(os.path.join(metrics_base_path, f"last_{dataset}.pkl"),
                        os.path.join(metrics_base_path, f"best_{dataset}.pkl"))

        trainer.test(
            model, model.get_dataloader(files["train"], config["batch_size"]))
        save_metrics("lowner_train")
        trainer.test(model,
                     model.get_dataloader(files["dev"], config["batch_size"]))
        save_metrics("lowner_dev")
        trainer.test(model,
                     model.get_dataloader(files["test"], config["batch_size"]))
        save_metrics("lowner_test")


if __name__ == "__main__":
    experiment01(
        "sent",
        [os.path.join(thesis_path, "data/multiconer/multiconer_sent.json")])
    experiment01(
        "lownergaz",
        [os.path.join(thesis_path, "data/mlowner/lowner_gazetteer.json")])
    experiment01("lownergaz_sent", [
        os.path.join(thesis_path, "data/multiconer/multiconer_sent.json"),
        os.path.join(thesis_path, "data/mlowner/lowner_gazetteer.json")
    ])
