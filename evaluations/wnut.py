import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from pipelines.evaluation import run_evaluation
from configs.asp_t5 import T5_BASE
from data_preparation.wnut import wnut_to_json

eval_dir_path = thesis_path + "/evaluations/test"

if not os.path.exists(eval_dir_path):
    os.makedirs(eval_dir_path)

wnut_to_json(
    "/home/loebbert/projects/thesis/data/wnut/wnut17train.conll",
    "/home/loebbert/projects/thesis/data/wnut/emerging.dev.conll",
    "/home/loebbert/projects/thesis/data/wnut/emerging.test.annotated")

run_evaluation(eval_dir_path=eval_dir_path,
               name="test",
               model_config=T5_BASE,
               dataset_files={
                   "train":
                   "/home/loebbert/projects/thesis/data/wnut/wnut_train.json",
                   "dev":
                   "/home/loebbert/projects/thesis/data/wnut/wnut_dev.json",
                   "test":
                   "/home/loebbert/projects/thesis/data/wnut/wnut_test.json",
                   "types":
                   "/home/loebbert/projects/thesis/data/wnut/wnut_types.json"
               },
               search_algorithm="bm25",
               search_topk=5,
               prepend_search_results=False,
               gold_database_source="dataset",
               error_database_size=0.5,
               data_type="gazetteers",
               input_processing="few-shot",
               use_labels=True,
               use_mentions=True,
               sampling="topk",
               sampling_value=1.0,
               beam_size=1,
               validate_on_test=False,
               num_runs=3)
