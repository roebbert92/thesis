import pickle
import sys
import os

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


def train_model(data_path: str, corpus: Corpus, seed: int):
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
    training_result = trainer.fine_tune(
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

    return training_result


if __name__ == "__main__":
    seeds = [2, 3]
    dataset = get_lowner_dataset()
    datapath = os.path.join("/home/loebbert/projects/thesis", "experiments",
                            "01_performance", "data")
    for seed in seeds:
        trained = False
        while not trained:
            try:
                train_model(datapath, dataset, seed)
                trained = True
            except RuntimeError:
                torch.cuda.empty_cache()