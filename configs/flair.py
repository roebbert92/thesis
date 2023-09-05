import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1]
)
sys.path.append(thesis_path)

FLAIR = {
    "lr": 5e-6,
    "max_epochs": 20,
    "max_length": 512,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.0,
    "locked_dropout_prob": 0.5,
    "hidden_dropout_prob": 0.0,
    "train_batch_size": 40,
    "eval_batch_size": 120,
    "accumulate_grad_batches": 1,
    "plm_name": "xlm-roberta-base",
    "dataset_prefix": "lowner_",
    "dataset_dir": os.path.join(thesis_path, "data", "mlowner"),
    "data_path": os.path.join(thesis_path, "experiments", "01_performance", "data"),
    "search_results_dir": None,
    "name": "flair",
}
FLAIR_GAZ = FLAIR.copy()
FLAIR_GAZ.update(
    {
        "search_results_dir": os.path.join(
            thesis_path,
            "experiments",
            "01_performance",
            "data",
            "01_search_results",
            "gaz",
        ),
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "train_search_dropout": 0.028260869565217374,
        "name": "flair_gaz",
    }
)
FLAIR_SENT = FLAIR.copy()
FLAIR_SENT.update(
    {
        "search_results_dir": os.path.join(
            thesis_path,
            "experiments",
            "01_performance",
            "data",
            "01_search_results",
            "sent",
        ),
        "sent_use_labels": True,
        "sent_use_mentions": True,
        "train_search_dropout": 0.21126587935893093,
        "name": "flair_sent",
    }
)

FLAIR_LOWNERGAZ = FLAIR.copy()
FLAIR_LOWNERGAZ.update(
    {
        "search_results_dir": os.path.join(
            thesis_path,
            "experiments",
            "01_performance",
            "data",
            "01_search_results",
            "lownergaz",
        ),
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "train_search_dropout": 0.028260869565217374,
        "name": "flair_lownergaz",
    }
)

FLAIR_LOWNERGAZ_GAZ = FLAIR.copy()
FLAIR_LOWNERGAZ_GAZ.update(
    {
        "search_results_dir": os.path.join(
            thesis_path,
            "experiments",
            "01_performance",
            "data",
            "01_search_results",
            "gaz",
        ),
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "train_search_dropout": 0.028260869565217374,
        "name": "flair_lownergaz_gaz",
    }
)

FLAIR_GAZ_SENT = FLAIR.copy()
FLAIR_GAZ_SENT.update(
    {
        "search_results_dir": os.path.join(
            thesis_path,
            "experiments",
            "01_performance",
            "data",
            "01_search_results",
            "gaz",
        ),
        "sent_use_labels": True,
        "sent_use_mentions": True,
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "train_search_dropout": 0.05492957746478871,
        "name": "flair_gaz",
    }
)

FLAIR_LOWNERGAZ_SENT = FLAIR.copy()
FLAIR_LOWNERGAZ_SENT.update(
    {
        "search_results_dir": os.path.join(
            thesis_path,
            "experiments",
            "01_performance",
            "data",
            "01_search_results",
            "lownergaz_sent",
        ),
        "sent_use_labels": True,
        "sent_use_mentions": True,
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "train_search_dropout": 0.05492957746478871,
        "name": "flair_lownergaz_sent",
    }
)
