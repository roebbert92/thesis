import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_preparation.wnut import wnut_to_json
from data_preprocessing.tokenize import tokenize_json
from pipelines.asp_training import run_experiment
from transformers import T5Tokenizer
from configs.asp_t5 import T5_BASE

wnut_to_json(
    "/home/loebbert/projects/thesis/data/wnut/wnut17train.conll",
    "/home/loebbert/projects/thesis/data/wnut/emerging.dev.conll",
    "/home/loebbert/projects/thesis/data/wnut/emerging.test.annotated")

tokenizer = T5Tokenizer.from_pretrained("t5-small", max_model_length=4096)

tokenize_json(tokenizer,
              "/home/loebbert/projects/thesis/data/wnut/wnut_train.json",
              "/home/loebbert/projects/thesis/data/wnut/wnut_dev.json",
              "/home/loebbert/projects/thesis/data/wnut/wnut_test.json",
              "/home/loebbert/projects/thesis/data/wnut/wnut_types.json")

config = T5_BASE
log_path = thesis_path + "/experiments/wnut/"
run_experiment(
    "wnut_asp_t5_base",
    "/home/loebbert/projects/thesis/data/wnut/wnut_train.t5-small.jsonlines",
    "/home/loebbert/projects/thesis/data/wnut/wnut_dev.t5-small.jsonlines",
    "/home/loebbert/projects/thesis/data/wnut/wnut_test.t5-small.jsonlines",
    "/home/loebbert/projects/thesis/data/wnut/wnut_types.json", log_path,
    config)
