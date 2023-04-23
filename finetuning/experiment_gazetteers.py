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

import pickle as pkl

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
config["name"] = "gazetteers"

best_configs = [
    copy.deepcopy(config), {
        'asp_hidden_dim': 629,
        'asp_dropout_rate': 0.3,
        'asp_init_std': 0.019999999999999976,
        'asp_activation': 'tanh',
        'plm_learning_rate': 4.999999999999997e-05,
        'task_learning_rate': 0.00029999999999999987,
        'adam_weight_decay': 0.1,
        'warmup_ratio': 0.21860103276831117,
        'use_labels': True,
        'use_mentions': False,
        'prepend_search_results': False,
        'filter_exact_match': False,
        'filter_same_document': False,
        'search_algorithm': 'ann',
        'search_topk': 10,
        'train_search_dropout': 0.0,
        'train_search_shuffle': False,
        'plm_pretrained_name_or_path': 't5-base',
        'plm_tokenizer_name': 't5-small',
        'model_max_length': 4096,
        'mention_start_token': '<m>',
        'mention_end_token': '</m>',
        'num_labels': 6,
        'max_nest_depth': 1,
        'beam_size': 1,
        'plm_scheduler': 'linear_with_warmup',
        'task_scheduler': 'linear_with_warmup',
        'adam_eps': 1e-08,
        'num_epochs': 20,
        'gradient_accumulation_steps': 1,
        'batch_size': 40,
        'train_len': 3394,
        'fused': True,
        'search_data_type': 'gazetteers',
        'seed': 42,
        'data_path': '/home/loebbert/projects/thesis/finetuning/tune',
        'name': 'gazetteers'
    }
]

config["asp_hidden_dim"] = tune.qrandint(100, 1000, 10)
config["asp_dropout_rate"] = tune.quniform(0.0, 0.5, 0.05)
config["asp_init_std"] = tune.quniform(5e-3, 0.5, 5e-3)
config["asp_activation"] = tune.choice(["relu", "gelu_fast", "tanh"])
config["beam_size"] = 1
config["use_labels"] = tune.choice([True, False])
config["use_mentions"] = tune.choice([True, False])
config["prepend_search_results"] = tune.choice([True, False])
config["filter_exact_match"] = tune.choice([True, False])
config["filter_same_document"] = tune.choice([True, False])
config["search_data_type"] = "gazetteers"
config["search_algorithm"] = tune.choice(["bm25", "ann", "ann+reranking"])
config["search_topk"] = tune.qrandint(1, 40)
config["seed"] = 42
config["train_search_dropout"] = tune.quniform(0.0, 1.0, 0.05)
config["train_search_shuffle"] = tune.choice([True, False])
config["plm_learning_rate"] = tune.quniform(5e-6, 5e-4, 5e-7)
config["task_learning_rate"] = tune.quniform(1e-5, 5e-3, 1e-5)
config["adam_weight_decay"] = tune.quniform(1e-3, 0.5, 5e-4)
config["warmup_ratio"] = tune.quniform(0.01, 0.99, 0.01)

param_space = {}
fixed_params = {}
for key, value in config.items():
    if isinstance(value, str) or isinstance(value, float) or isinstance(
            value, int) or isinstance(value, bool) or value is None:
        fixed_params[key] = value
        for c in best_configs:
            del c[key]
    else:
        param_space[key] = value
        print(key)

reporter = tune.CLIReporter(
    # parameter_columns=["batch_size"],
    metric_columns=["val_f1", "training_iteration"])

ng_search = NevergradSearch(
    optimizer=ng.optimizers.PortfolioDiscreteOnePlusOne,
    metric="val_f1",
    mode="max",
    points_to_evaluate=best_configs)

method = tune.with_resources(tune.with_parameters(run_tune_training,
                                                  fixed_params=fixed_params),
                             resources={
                                 "cpu": 12,
                                 "gpu": 1
                             })

tuner = tune.Tuner(method,
                   param_space=param_space,
                   tune_config=tune.TuneConfig(metric="val_f1",
                                               mode="max",
                                               search_alg=ng_search,
                                               num_samples=-1,
                                               time_budget_s=4 * 60 * 60),
                   run_config=air.RunConfig(local_dir=config["data_path"],
                                            name="gazetteers",
                                            progress_reporter=reporter))
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result().config)

with open(os.path.join(thesis_path, "finetuning", "gazetteers_result.pkl"),
          "wb") as file:
    pkl.dump(results, file)