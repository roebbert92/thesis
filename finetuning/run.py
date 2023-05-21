import sys
import os
from typing import Optional

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from ray import tune, air
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.schedulers import ASHAScheduler

import nevergrad as ng

import pickle as pkl

from finetuning.t5_asp import t5_asp_configs, run_t5_asp_training
from finetuning.t5_asp_fetahugaz import t5_asp_fetahugaz_configs, run_t5_asp_fetahugaz_training


def finetune(name, config, best_configs, tune_training_method,
             training_budget_in_h):
    param_space = {}
    fixed_params = {}
    for key, value in config.items():
        if isinstance(value, str) or isinstance(value, float) or isinstance(
                value, int) or isinstance(value, bool) or value is None:
            fixed_params[key] = value
            for c in best_configs:
                if key in c:
                    del c[key]
        else:
            param_space[key] = value
            print(key)

    for c in best_configs:
        for key in list(c.keys()):
            if key not in config:
                del c[key]

    reporter = tune.CLIReporter(
        # parameter_columns=["batch_size"],
        metric_columns=["val_loss", "training_iteration"])

    ng_search = NevergradSearch(
        optimizer=ng.optimizers.RecombiningPortfolioDiscreteOnePlusOne,
        metric="val_loss",
        mode="min",
        points_to_evaluate=best_configs)

    method = tune.with_resources(tune.with_parameters(
        tune_training_method, fixed_params=fixed_params),
                                 resources={
                                     "cpu": 12,
                                     "gpu": 1
                                 })

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='val_loss',
        mode='min',
        max_t=100,
        grace_period=5,
        reduction_factor=4,
        brackets=5,
    )

    tuner = tune.Tuner(method,
                       param_space=param_space,
                       tune_config=tune.TuneConfig(
                           scheduler=scheduler,
                           search_alg=ng_search,
                           num_samples=-1,
                           time_budget_s=training_budget_in_h * 60 * 60),
                       run_config=air.RunConfig(local_dir=config["data_path"],
                                                name=name,
                                                progress_reporter=reporter))
    results = tuner.fit()
    print("Best val_loss hyperparameters found were: ",
          results.get_best_result("val_loss", "min", "all").config)

    with open(os.path.join(thesis_path, "finetuning", f"{name}_result.pkl"),
              "wb") as file:
        pkl.dump(results, file)


# t5_asp_config = t5_asp_configs()
# finetune("t5_asp", t5_asp_config[0], t5_asp_config[1], run_t5_asp_training, 5)

t5_asp_config = t5_asp_configs()
finetune("t5_asp", t5_asp_config[0], t5_asp_config[1], run_t5_asp_training, 5)

t5_asp_fetahugaz_config = t5_asp_fetahugaz_configs()
finetune("t5_asp_fetahugaz", t5_asp_fetahugaz_config[0],
         t5_asp_fetahugaz_config[1], run_t5_asp_fetahugaz_training, 5)