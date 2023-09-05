import sys
import os
from typing import Optional

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1]
)
sys.path.append(thesis_path)

from ray import tune, air
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.schedulers import ASHAScheduler

import nevergrad as ng

import pickle as pkl

# from hyperparameter_tuning.t5_asp import t5_asp_configs, run_t5_asp_training
# from hyperparameter_tuning.t5_asp_lownergaz import t5_asp_lownergaz_configs, run_t5_asp_lownergaz_training
# from hyperparameter_tuning.t5_asp_gaz import t5_asp_gaz_configs, t5_asp_gaz_configs, run_t5_asp_gaz_training
# from hyperparameter_tuning.t5_asp_sent import t5_asp_sent_configs, t5_asp_sent_configs, run_t5_asp_sent_training
# from hyperparameter_tuning.t5_asp_gaz_sent import t5_asp_gaz_sent_configs, t5_asp_gaz_sent_configs, run_t5_asp_gaz_sent_training
# from hyperparameter_tuning.t5_asp_lownergaz_sent import t5_asp_lownergaz_sent_configs, run_t5_asp_lownergaz_sent_training
# from hyperparameter_tuning.t5_asp_lownergaz_sent_wnut import (
#     wnut_t5_asp_lownergaz_sent_configs,
#     run_wnut_t5_asp_lownergaz_sent_training,
# )
# from hyperparameter_tuning.t5_asp_lownergaz_sent_wnut_pretrained import (
#     best_pretrained_wnut_t5_asp_lownergaz_sent_configs,
#     worst_pretrained_wnut_t5_asp_lownergaz_sent_configs,
#     run_pretrained_wnut_t5_asp_lownergaz_sent_training,
# )
# from hyperparameter_tuning.t5_asp_wnut import (
#     wnut_t5_asp_configs,
#     run_wnut_t5_asp_training,
# )
# from hyperparameter_tuning.t5_asp_wnut_pretrained import (
#     run_best_wnut_t5_asp_training,
#     best_pretrained_wnut_t5_asp_configs,
# )
from hyperparameter_tuning.flair_roberta import (
    flair_configs,
    flair_gaz_configs,
    flair_gaz_sent_configs,
    flair_lownergaz_configs,
    flair_lownergaz_gaz_configs,
    flair_lownergaz_sent_configs,
    flair_sent_configs,
    run_training,
)


def tune_hyperparameters(
    name, config, best_configs, tune_training_method, training_budget_in_h
):
    param_space = {}
    fixed_params = {}
    for key, value in config.items():
        if (
            isinstance(value, str)
            or isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, bool)
            or value is None
        ):
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
        metric_columns=["val_f1", "training_iteration"]
    )

    budget = training_budget_in_h * 20
    # if name == "t5_asp":
    #     budget = 140

    ng_search = NevergradSearch(
        optimizer=ng.optimizers.ScrHammersleySearch,
        optimizer_kwargs={"budget": budget},
        metric="val_f1",
        mode="max",
        points_to_evaluate=best_configs,
    )

    method = tune.with_resources(
        tune.with_parameters(tune_training_method, fixed_params=fixed_params),
        resources={"cpu": 12, "gpu": 1},
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="val_f1",
        mode="max",
        max_t=100,
        grace_period=2,
        reduction_factor=2,
        brackets=1,
    )

    tuner = tune.Tuner(
        method,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=ng_search,
            num_samples=-1,
            time_budget_s=training_budget_in_h * 60 * 60,
        ),
        run_config=air.RunConfig(
            local_dir=config["data_path"], name=name, progress_reporter=reporter
        ),
    )
    results = tuner.fit()
    print(
        "Best hyperparameters found were: ",
        results.get_best_result("val_f1", "max", "all").config,
    )

    with open(
        os.path.join(thesis_path, "hyperparameter_tuning", f"{name}_result.pkl"), "wb"
    ) as file:
        pkl.dump(results, file)


def tune_wnut_hyperparameters(
    name, config, best_configs, tune_training_method, training_budget_in_h
):
    param_space = {}
    fixed_params = {}
    for key, value in config.items():
        if (
            isinstance(value, str)
            or isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, bool)
            or value is None
        ):
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
        metric_columns=["f1", "training_iteration"]
    )

    training_budget_sec = training_budget_in_h * 60 * 60
    time_per_epoch_sec = 35
    if "num_epochs" in param_space:
        min_epochs = (
            param_space["num_epochs"].lower
            + (param_space["num_epochs"].upper - param_space["num_epochs"].lower) / 2
        )
    else:
        min_epochs = fixed_params["num_epochs"]
    budget = int(training_budget_sec * 1.5 / (min_epochs * time_per_epoch_sec)) + 1
    # find good range:
    # optimizer = ng.optimizers.DifferentialEvolution(crossover="twopoints",
    #                                                 high_speed=True)
    # optimizer_kwargs = None
    # points_to_evaluate = best_configs
    # narrow search
    optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint
    optimizer_kwargs = {"budget": budget}
    points_to_evaluate = None
    ng_search = NevergradSearch(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        metric="f1",
        mode="max",
        points_to_evaluate=points_to_evaluate,
    )

    method = tune.with_resources(
        tune.with_parameters(tune_training_method, fixed_params=fixed_params),
        resources={"cpu": 12, "gpu": 1},
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="f1",
        mode="max",
        max_t=100,
        grace_period=4,
        reduction_factor=2,
        brackets=1,
    )

    tuner = tune.Tuner(
        method,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=ng_search,
            num_samples=-1,
            time_budget_s=training_budget_sec,
        ),
        run_config=air.RunConfig(
            local_dir=config["data_path"], name=name, progress_reporter=reporter
        ),
    )
    results = tuner.fit()
    print(
        "Best hyperparameters found were: ",
        results.get_best_result("f1", "max", "all").config,
    )

    with open(
        os.path.join(thesis_path, "hyperparameter_tuning", f"{name}_result.pkl"), "wb"
    ) as file:
        pkl.dump(results, file)


def tune_flair_hyperparameters(
    config, best_configs, tune_training_method, training_budget_in_h
):
    name = config["name"]
    param_space = {}
    fixed_params = {}
    for key, value in config.items():
        if (
            isinstance(value, str)
            or isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, bool)
            or value is None
        ):
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
        metric_columns=["f1", "training_iteration"]
    )

    training_budget_sec = training_budget_in_h * 60 * 60
    # time_per_epoch_sec = 35
    # if "num_epochs" in param_space:
    #     min_epochs = (
    #         param_space["num_epochs"].lower
    #         + (param_space["num_epochs"].upper - param_space["num_epochs"].lower) / 2
    #     )
    # else:
    #     min_epochs = fixed_params["num_epochs"]
    budget = training_budget_in_h * 20
    # find good range:
    optimizer = ng.optimizers.DifferentialEvolution(
        crossover="twopoints", high_speed=True
    )
    optimizer_kwargs = None
    points_to_evaluate = best_configs
    # narrow search
    # optimizer = ng.optimizers.ScrHammersleySearchPlusMiddlePoint
    # optimizer_kwargs = {"budget": budget}
    # points_to_evaluate = None
    ng_search = NevergradSearch(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        metric="f1",
        mode="max",
        points_to_evaluate=points_to_evaluate,
    )

    method = tune.with_resources(
        tune.with_parameters(tune_training_method, fixed_params=fixed_params),
        resources={"cpu": 12, "gpu": 1},
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="f1",
        mode="max",
        max_t=100,
        grace_period=4,
        reduction_factor=2,
        brackets=1,
    )

    tuner = tune.Tuner(
        method,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=ng_search,
            num_samples=-1,
            time_budget_s=training_budget_sec,
        ),
        run_config=air.RunConfig(
            local_dir=config["data_path"], name=name, progress_reporter=reporter
        ),
    )
    results = tuner.fit()
    print(
        "Best hyperparameters found were: ",
        results.get_best_result("f1", "max", "all").config,
    )

    with open(
        os.path.join(thesis_path, "hyperparameter_tuning", f"{name}_result.pkl"), "wb"
    ) as file:
        pkl.dump(results, file)


training_budget = 4

# wnut = wnut_t5_asp_lownergaz_sent_configs()
# tune_wnut_hyperparameters("wnut_t5_asp_lownergaz_sent", wnut[0], wnut[1],
#                           run_wnut_t5_asp_lownergaz_sent_training,
#                           training_budget)

# wnut_vanilla = wnut_t5_asp_configs()
# tune_wnut_hyperparameters("vanilla_wnut_t5_asp", wnut_vanilla[0],
#                           wnut_vanilla[1], run_wnut_t5_asp_training,
#                           training_budget)
# wnut_vanilla_pretrained = best_pretrained_wnut_t5_asp_configs()
# tune_wnut_hyperparameters("best_vanilla_wnut_t5_asp",
#                           wnut_vanilla_pretrained[0],
#                           wnut_vanilla_pretrained[1],
#                           run_best_wnut_t5_asp_training, training_budget)

# wnut_worst = worst_pretrained_wnut_t5_asp_lownergaz_sent_configs(
#     "/home/loebbert/projects/thesis/experiments/02_content/data/seed_2/03_checkpoints/size_4000/error_ratio_15/last.ckpt"
# )
# tune_wnut_hyperparameters("worst_pretrained_wnut_t5_asp", wnut_worst[0],
#                           wnut_worst[1],
#                           run_pretrained_wnut_t5_asp_lownergaz_sent_training,
#                           training_budget)
# wnut_best = best_pretrained_wnut_t5_asp_lownergaz_sent_configs(
#     "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/03_checkpoints/t5_asp_lownergaz_sent/last.ckpt"
# )
# tune_wnut_hyperparameters(
#     "best_pretrained_wnut_t5_asp",
#     wnut_best[0],
#     wnut_best[1],
#     run_pretrained_wnut_t5_asp_lownergaz_sent_training,
#     training_budget,
# )
#
# wnut = wnut_t5_asp_lownergaz_sent_configs()
# tune_wnut_hyperparameters(
#     "wnut_t5_asp_lownergaz_sent",
#     wnut[0],
#     wnut[1],
#     run_wnut_t5_asp_lownergaz_sent_training,
#     training_budget,
# )#

flair = flair_configs()
tune_flair_hyperparameters(flair[0], flair[1], run_training, training_budget)

flair_gaz = flair_gaz_configs()
tune_flair_hyperparameters(flair_gaz[0], flair_gaz[1], run_training, training_budget)

flair_sent = flair_sent_configs()
tune_flair_hyperparameters(flair_sent[0], flair_sent[1], run_training, training_budget)

flair_lownergaz = flair_lownergaz_configs()
tune_flair_hyperparameters(
    flair_lownergaz[0], flair_lownergaz[1], run_training, training_budget
)

flair_gaz_sent = flair_gaz_sent_configs()
tune_flair_hyperparameters(
    flair_gaz_sent[0], flair_gaz_sent[1], run_training, training_budget
)

flair_lownergaz_gaz = flair_lownergaz_gaz_configs()
tune_flair_hyperparameters(
    flair_lownergaz_gaz[0], flair_lownergaz_gaz[1], run_training, training_budget
)

flair_lownergaz_sent = flair_lownergaz_sent_configs()
tune_flair_hyperparameters(
    flair_lownergaz_sent[0], flair_lownergaz_sent[1], run_training, training_budget
)

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
