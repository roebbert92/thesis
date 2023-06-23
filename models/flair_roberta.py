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
        model='roberta-large',
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
                                     "03_checkpoints")
    training_result = trainer.fine_tune(
        model_output_path,
        seed=seed,
        device="cuda:0",
        use_amp=True,
        learning_rate=5.0e-6,
        mini_batch_size=8,
        eval_batch_size=24,
        max_epochs=10,
        mini_batch_chunk_size=1,
        param_selection_mode=True,
        monitor_train=True,
        save_final_model=True,
        main_evaluation_metric=("macro avg", "f1-score"),
        use_tensorboard=True,
        tensorboard_log_dir=os.path.join(data_path, "lightning_logs"),
        metrics_for_tensorboard=[("macro avg", "f1-score"),
                                 ("macro avg", "precision"),
                                 ("macro avg", "recall")])

    return training_result