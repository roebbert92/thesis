import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1]
)
sys.path.append(thesis_path)

FLAIR = {
    "max_epochs": 20,
    "max_length": 512,
    "adam_epsilon": 1e-8,
    "hidden_dropout_prob": 0.2,
    "locked_dropout_prob": 0.0,
    "lr": 2.2346020761245675e-05,
    "weight_decay": 0.027590027700831025,
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
        "hidden_dropout_prob": 0.2,
        "locked_dropout_prob": 0.5,
        "lr": 2.061764705882353e-05,
        "train_search_dropout": 0.1368421052631579,
        "weight_decay": 0.026086956521739132,
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
        "hidden_dropout_prob": 0.1,
        "locked_dropout_prob": 0.5,
        "lr": 2.355588235294118e-05,
        "train_search_dropout": 0.34210526315789475,
        "weight_decay": 0.020869565217391303,
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
        "hidden_dropout_prob": 0.1,
        "locked_dropout_prob": 0.3,
        "lr": 3.392612456747405e-05,
        "train_search_dropout": 0.06301939058171747,
        "weight_decay": 0.015652173913043476,
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
            "lownergaz_gaz",
        ),
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "hidden_dropout_prob": 0.2,
        "locked_dropout_prob": 0.0,
        "lr": 4.913581314878893e-05,
        "train_search_dropout": 0.16024930747922436,
        "weight_decay": 0.006521739130434783,
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
            "gaz_sent",
        ),
        "sent_use_labels": True,
        "sent_use_mentions": True,
        "gaz_use_labels": True,
        "gaz_use_mentions": True,
        "hidden_dropout_prob": 0.2,
        "locked_dropout_prob": 0.5,
        "lr": 4.1185294117647066e-05,
        "train_search_dropout": 0.5131578947368421,
        "weight_decay": 0.02347826086956522,
        "name": "flair_gaz_sent",
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
        "hidden_dropout_prob": 0.2,
        "locked_dropout_prob": 0.3,
        "lr": 4.706176470588235e-05,
        "train_search_dropout": 0.03421052631578948,
        "weight_decay": 0.024782608695652172,
        "name": "flair_lownergaz_sent",
    }
)
