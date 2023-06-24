import gc
import pickle
import sys
import os
from typing import List

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from data_preparation.lowner import wnut_types
import lightning.pytorch as pl
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch
from torch.utils.data import Dataset

from models.metrics import ASPMetrics
from data_preparation.flair_evaluation import read_flair_predictions_to_asp, asp_dataset_to_asp_predictions


def get_lowner_dataset():
    columns = {0: 'text', 1: '', 2: '', 3: 'ner'}
    lowner_folder = os.path.join(thesis_path, "data", "mlowner")
    corpus: Corpus = ColumnCorpus(lowner_folder,
                                  columns,
                                  train_file="train_lower.txt",
                                  dev_file="dev_lower.txt",
                                  test_file="test_lower.txt",
                                  label_name_map=wnut_types)
    return corpus


def train_model(data_path: str, corpus: Corpus, seed: int,
                evaluation_dataset_names: List[str]):
    label_type = "ner"
    label_dict = corpus.make_label_dictionary(label_type=label_type,
                                              add_unk=False)
    embeddings = TransformerWordEmbeddings(
        model='xlm-roberta-large',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
    )
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type='ner',
        tag_format="BIO",
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )
    trainer = ModelTrainer(tagger, corpus)
    model_output_path = os.path.join(data_path, f"seed_{seed}",
                                     "03_checkpoints", "flair_roberta")
    os.makedirs(model_output_path, exist_ok=True)
    tensorboard_logdir = os.path.sep + os.path.join(
        os.path.join(*data_path.split(os.path.sep)[:-1]), "lightning_logs",
        f"{seed}_flair_roberta")
    os.makedirs(tensorboard_logdir, exist_ok=True)
    # seed
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    pl.seed_everything(seed)
    trainer.fine_tune(
        model_output_path,
        #device="cuda:0",
        use_amp=True,
        learning_rate=5e-6,
        mini_batch_size=40,
        eval_batch_size=120,
        max_epochs=20,  # 20
        mini_batch_chunk_size=15,
        num_workers=3,
        optimizer=AdamW,
        scheduler=OneCycleLR,  # type: ignore
        embeddings_storage_mode='none',
        weight_decay=0.,
        shuffle_first_epoch=True,
        param_selection_mode=False,
        use_final_model_for_eval=False,
        monitor_train=False,
        monitor_test=False,
        save_final_model=True,
        main_evaluation_metric=("macro avg", "f1-score"),
        use_tensorboard=True,
        tensorboard_log_dir=tensorboard_logdir,
        tensorboard_comment=f"{seed}_flair_roberta",
        metrics_for_tensorboard=[("macro avg", "f1-score"),
                                 ("macro avg", "precision"),
                                 ("macro avg", "recall")])
    del tagger
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # metrics
    metrics_base_path = os.path.join(data_path, f"seed_{seed}", "04_metrics",
                                     "flair_roberta")

    # get metrics for last ckpt
    last_ckpt_path = os.path.join(model_output_path, "final-model.pt")
    get_asp_metrics_from_flair(metrics_base_path, evaluation_dataset_names,
                               corpus, "last", last_ckpt_path)

    # get metrics for best ckpt
    best_ckpt_path = os.path.join(model_output_path, "best-model.pt")
    get_asp_metrics_from_flair(metrics_base_path, evaluation_dataset_names,
                               corpus, "best", best_ckpt_path)


def get_asp_metrics_from_flair(metrics_base_path: str, dataset_names: list,
                               flair_dataset: Corpus, ckpt_name: str,
                               ckpt_path: str):

    tagger = SequenceTagger.load(ckpt_path)

    os.makedirs(metrics_base_path, exist_ok=True)

    for dataset_name in dataset_names:
        # produce prediction file
        dataset_part = dataset_name.split("_")[-1]
        temp_path = os.path.join(metrics_base_path, "flair_prediction.txt")
        flair_data = flair_dataset.train
        if dataset_part == "dev":
            flair_data = flair_dataset.dev
        elif dataset_part == "test":
            flair_data = flair_dataset.test
        tagger.evaluate(
            flair_data,  # type: ignore
            "ner",
            out_path=temp_path,
            return_loss=False,
            mini_batch_size=120)
        # init metrics
        metrics = ASPMetrics()
        metrics.predictions = asp_dataset_to_asp_predictions(
            read_flair_predictions_to_asp(dataset_name, temp_path))
        os.remove(temp_path)
        # save metrics
        metrics_path = os.path.join(metrics_base_path,
                                    f"{ckpt_name}_{dataset_name}.pkl")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
    del tagger
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    seeds = [1, 2, 3]
    dataset = get_lowner_dataset()
    datapath = os.path.join("/home/loebbert/projects/thesis", "experiments",
                            "01_performance", "data")
    dataset_names = ["lowner_train", "lowner_dev", "lowner_test"]
    for seed in seeds:
        trained = False
        while not trained:
            try:
                train_model(datapath, dataset, seed, dataset_names)
                trained = True
            except RuntimeError:
                torch.cuda.empty_cache()