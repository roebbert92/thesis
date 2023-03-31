import sys
import os
import json

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from pipelines.asp_training import run_experiment
from configs.asp_t5 import T5_BASE

experiments = {
    "huggingface_ours":
    "/home/loebbert/projects/thesis/data/conll03/huggingface/ours",
    "eth_ours": "/home/loebbert/projects/thesis/data/conll03/eth/ours",
    "huggingface_asp":
    "/home/loebbert/projects/thesis/data/conll03/huggingface/asp",
    "eth_asp": "/home/loebbert/projects/thesis/data/conll03/eth/asp",
}

for name, dir_path in experiments.items():
    log_path = thesis_path + "/experiments/conll03/" + name
    config = T5_BASE
    if name.endswith("asp"):
        config["batch_size"] = 1
        config["gradient_accumulation_steps"] = 1
    try:
        result = run_experiment(
            name,
            os.path.join(dir_path, "conll03_train.t5-small.jsonlines"),
            os.path.join(dir_path, "conll03_dev.t5-small.jsonlines"),
            os.path.join(dir_path, "conll03_test.t5-small.jsonlines"),
            os.path.join(dir_path, "conll03_types.json"),
            logger_dir_path=log_path,
            config=config)
    except Exception as e:
        result = str(e)

    with open(os.path.join(log_path, "result.json"), "w",
              encoding="utf-8") as file:
        json.dump({"result": result}, file)
