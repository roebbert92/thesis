# Nevergrad optimizer: PortfolioDiscreteOnePlusOne
# Seed = 42
import copy
import sys
import os
from typing import Optional

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from configs.asp_t5 import T5_BASE

from ray import tune, air
from ray.tune.search.nevergrad import NevergradSearch

from finetuning.training import run_tune_training

import nevergrad as ng

config = T5_BASE
config["asp_hidden_dim"] = 150
config["asp_dropout_rate"] = 0.3
config["asp_init_std"] = 0.02
config["asp_activation"] = "relu"
config["beam_size"] = 1
config["use_labels"] = True
config["use_mentions"] = True
config["prepend_search_results"] = False
config["filter_exact_match"] = False
config["filter_same_document"] = False
config["search_data_type"] = "gazetteers"
config["search_algorithm"] = "bm25"
config["search_topk"] = 10
config["seed"] = 42
config["train_search_dropout"] = 0.0
config["train_search_shuffle"] = False

config["data_path"] = os.path.join(thesis_path, "finetuning", "tune")
config["name"] = "test"

current_best_config = copy.deepcopy(config)

config["asp_hidden_dim"] = tune.choice([150, 250, 4096])
config["asp_dropout_rate"] = tune.choice([0.3, 0.2, 0.1])
config["asp_init_std"] = tune.choice([0.02, 0.2, 0.04, 0.01])
config["asp_activation"] = tune.choice(["relu", "linear", "gelu_fast"])
config["beam_size"] = 1
config["use_labels"] = tune.choice([True, False])
config["use_mentions"] = tune.choice([True, False])
config["prepend_search_results"] = tune.choice([True, False])
config["filter_exact_match"] = tune.choice([True, False])
config["filter_same_document"] = tune.choice([True, False])
config["search_data_type"] = "gazetteers"
config["search_algorithm"] = tune.choice(["bm25", "ann", "ann+reranking"])
config["search_topk"] = tune.choice([5, 10, 20])
config["seed"] = 42
config["train_search_dropout"] = tune.choice([0.0, 0.1, 0.2])
config["train_search_shuffle"] = tune.choice([True, False])

reporter = tune.CLIReporter(
    # parameter_columns=["batch_size"],
    metric_columns=["val_f1", "training_iteration"])

ng_search = NevergradSearch(
    optimizer=ng.optimizers.PortfolioDiscreteOnePlusOne,
    metric="val_f1",
    mode="max",
    points_to_evaluate=[current_best_config])

tuner = tune.Tuner(tune.with_resources(run_tune_training,
                                       resources={
                                           "cpu": 12,
                                           "gpu": 1
                                       }),
                   param_space=config,
                   tune_config=tune.TuneConfig(search_alg=ng_search,
                                               num_samples=-1,
                                               time_budget_s=6 * 60 * 60),
                   run_config=air.RunConfig(local_dir=config["data_path"],
                                            name="gazetteers",
                                            progress_reporter=reporter))
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result().config)