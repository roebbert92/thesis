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

#from hyperparameter_tuning.t5_asp import t5_asp_configs, run_t5_asp_training
#from hyperparameter_tuning.t5_asp_lownergaz import t5_asp_lownergaz_configs, run_t5_asp_lownergaz_training
#from hyperparameter_tuning.t5_asp_gaz import t5_asp_gaz_configs, t5_asp_gaz_configs, run_t5_asp_gaz_training
#from hyperparameter_tuning.t5_asp_sent import t5_asp_sent_configs, t5_asp_sent_configs, run_t5_asp_sent_training
#from hyperparameter_tuning.t5_asp_gaz_sent import t5_asp_gaz_sent_configs, t5_asp_gaz_sent_configs, run_t5_asp_gaz_sent_training
#from hyperparameter_tuning.t5_asp_lownergaz_sent import t5_asp_lownergaz_sent_configs, run_t5_asp_lownergaz_sent_training
from hyperparameter_tuning.t5_asp_lownergaz_sent_wnut import wnut_t5_asp_lownergaz_sent_configs, run_wnut_t5_asp_lownergaz_sent_training


def tune_hyperparameters(name, config, best_configs, tune_training_method,
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
        metric_columns=["val_f1", "training_iteration"])

    budget = 60
    # if name == "t5_asp":
    #     budget = 140

    ng_search = NevergradSearch(optimizer=ng.optimizers.ScrHammersleySearch,
                                optimizer_kwargs={"budget": 60},
                                metric="val_f1",
                                mode="max",
                                points_to_evaluate=best_configs)

    method = tune.with_resources(tune.with_parameters(
        tune_training_method, fixed_params=fixed_params),
                                 resources={
                                     "cpu": 12,
                                     "gpu": 1
                                 })

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='val_f1',
        mode='max',
        max_t=100,
        grace_period=2,
        reduction_factor=2,
        brackets=1,
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
    print("Best hyperparameters found were: ",
          results.get_best_result("val_f1", "max", "all").config)

    with open(
            os.path.join(thesis_path, "hyperparameter_tuning",
                         f"{name}_result.pkl"), "wb") as file:
        pkl.dump(results, file)


def tune_wnut_hyperparameters(name, config, best_configs, tune_training_method,
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
        metric_columns=["f1", "training_iteration"])

    budget = 40
    # if name == "t5_asp":
    #     budget = 140

    ng_search = NevergradSearch(
        optimizer=ng.optimizers.ScrHammersleySearch,
        optimizer_kwargs={"budget": budget},
        metric="f1",
        mode="max",
        #points_to_evaluate=best_configs
    )

    method = tune.with_resources(tune.with_parameters(
        tune_training_method, fixed_params=fixed_params),
                                 resources={
                                     "cpu": 12,
                                     "gpu": 1
                                 })

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='f1',
        mode='max',
        max_t=100,
        grace_period=2,
        reduction_factor=2,
        brackets=1,
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
    print("Best hyperparameters found were: ",
          results.get_best_result("f1", "max", "all").config)

    with open(
            os.path.join(thesis_path, "hyperparameter_tuning",
                         f"{name}_result.pkl"), "wb") as file:
        pkl.dump(results, file)


training_budget = 3

wnut = wnut_t5_asp_lownergaz_sent_configs()
tune_wnut_hyperparameters("wnut_t5_asp_lownergaz_sent", wnut[0], wnut[1],
                          run_wnut_t5_asp_lownergaz_sent_training,
                          training_budget)
# t5_asp_gaz_sent_config = t5_asp_gaz_sent_configs()
# tune_hyperparameters("t5_asp_gaz_sent", t5_asp_gaz_sent_config[0],
#                      t5_asp_gaz_sent_config[1], run_t5_asp_gaz_sent_training,
#                      training_budget)
# t5_asp_sent_config = t5_asp_sent_configs()
# tune_hyperparameters("t5_asp_sent", t5_asp_sent_config[0],
#                      t5_asp_sent_config[1], run_t5_asp_sent_training,
#                      training_budget)
# t5_asp_gaz_config = t5_asp_gaz_configs()
# tune_hyperparameters("t5_asp_gaz", t5_asp_gaz_config[0], t5_asp_gaz_config[1],
#                      run_t5_asp_gaz_training, training_budget)
# t5_asp_lownergaz_config = t5_asp_lownergaz_configs()
# tune_hyperparameters("t5_asp_lownergaz", t5_asp_lownergaz_config[0],
#                      t5_asp_lownergaz_config[1], run_t5_asp_lownergaz_training,
#                      training_budget)
# t5_asp_config = t5_asp_configs()
# tune_hyperparameters("t5_asp", t5_asp_config[0], t5_asp_config[1],
#                      run_t5_asp_training, training_budget)
