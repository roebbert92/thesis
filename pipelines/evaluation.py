from argparse import ArgumentError
from collections import defaultdict
import json
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from tqdm import tqdm
import lightning.fabric.utilities.seed as pl_seed
import shutil
from haystack import Pipeline, Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever, SentenceTransformersRanker, BM25Retriever

from data_preprocessing.tokenize import tokenize_json, tokenize_database_json
from models.asp_t5 import get_tokenizer
from pipelines.asp_training import run_experiment


def run_evaluation(
    eval_dir_path: str,
    name: str,
    model_config: dict,
    dataset_files: dict,
    search_algorithm: str,  # bm25, ann, ann+ranking
    search_topk: int,  # > 0
    prepend_search_results:
    bool,  # False query [SEP] results; True results [SEP] query
    gold_database_source: str,  # lowner, dataset
    error_database_size: float,  # 0...1 relative to gold_database_size
    data_type: str,  # sentences, gazetteers
    input_processing: str,  # fusion-in-decoder, few-shot
    use_labels: bool,  # :<label>
    use_mentions: bool,  # <m></m>
    sampling: str,  # topk, nucleus
    sampling_value: float,  # > 0
    beam_size: int,  # 1 = greedy search; > 0
    validate_on_test:
    bool,  # False for validating on devset, True for validating on testset
    num_runs: int  # > 1; how many runs
):
    model_metrics = defaultdict(dict)
    database_metrics = defaultdict(dict)
    # 0. loop over num_runs
    for run_id in tqdm(range(num_runs), desc="Runs"):
        run_dir_path = os.path.join(eval_dir_path, "run_" + str(run_id))
        # 0.1 lightning seed everything with new random seed (clear environment variable PL_GLOBAL_SEED)
        if "PL_GLOBAL_SEED" in os.environ:
            del os.environ["PL_GLOBAL_SEED"]
        seed = pl_seed.seed_everything()

        # 0.2 Copy dataset files to eval_dir_path/run_id/dataset
        files = {}
        for key, file_path in dataset_files.items():
            files[key] = os.path.join(run_dir_path, "dataset",
                                      os.path.basename(file_path))
            dir_name = os.path.dirname(files[key])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            shutil.copyfile(file_path, files[key])

        tokenizer = get_tokenizer(model_config)
        # 1. Train baseline
        # 1.1 Tokenize json
        tokenize_json(tokenizer, files["train"], files["dev"], files["test"],
                      files["types"])
        for key, file_path in list(files.items()):
            if key != "types":
                files["tokenized_" + key] = file_path.split(
                    ".")[0] + "." + tokenizer.name_or_path + ".jsonlines"
        # 1.2 Run experiment
        # 1.2.1 collect false positives from train + dev
        # (baseline_train_result, baseline_dev_result, baseline_test_result,
        #  baseline_train_false_positives, baseline_dev_false_positives,
        #  _) = run_experiment(
        #      name=name,
        #      tokenized_train_data_file=files["tokenized_train"],
        #      tokenized_dev_data_file=files["tokenized_dev"],
        #      tokenized_test_data_file=files["tokenized_train"],
        #      type_data_file=files["types"],
        #      logger_dir_path=run_dir_path,
        #      config=model_config,
        #      validate_on_test=validate_on_test)
        # false_positives = baseline_train_false_positives | baseline_dev_false_positives
        # # 1.2.2 report metrics
        # model_metrics[run_id]["baseline"]["train_f1"] = baseline_train_result[
        #     "val_f1_epoch"]  # type: ignore
        # model_metrics[run_id]["baseline"]["dev_f1"] = baseline_dev_result[
        #     "val_f1_epoch"]  # type: ignore
        # if baseline_test_result is not None:
        #     model_metrics[run_id]["baseline"][
        #         "test_f1"] = baseline_test_result[
        #             "test_f1_epoch"]  # type: ignore

    # 2. Prepare gold database
    # gold_document_store = name + "gold" + gold_database_source + seed
        gold_document_store = ElasticsearchDocumentStore(
            index=name + "_gold_" + gold_database_source + "_" + str(seed))
        gold_search = Pipeline()
        if search_algorithm == "bm25":
            bm25_retriever = BM25Retriever(gold_document_store,
                                           top_k=search_topk)
            gold_search.add_node(component=bm25_retriever,
                                 name="BM25Retriever",
                                 inputs=["Query"])
        elif search_algorithm.startswith("ann"):
            ann_retriever = EmbeddingRetriever(
                document_store=gold_document_store,
                embedding_model="sentence-transformers/all-mpnet-base-v2",
                model_format="sentence_transformers",
                top_k=search_topk)
            gold_search.add_node(component=ann_retriever,
                                 name="ANNRetriever",
                                 inputs=["Query"])
            if search_algorithm.endswith("ranking"):
                ranker = SentenceTransformersRanker(
                    model_name_or_path=
                    "sentence-transformers/msmarco-bert-base-dot-v5",
                    top_k=search_topk)
                gold_search.add_node(component=ranker,
                                     name="Ranker",
                                     inputs=["ANNRetriever"])

        if len(gold_search.components) == 0:
            raise Exception(
                "Argument error: search_algorithm - must be: bm25 | ann | ann+ranking, but is: "
                + search_algorithm)

        # 2.1 Load dataset jsons
        # 2.1 Load into search engine if gold_document_store does not exist
        if gold_document_store.get_document_count() == 0:
            documents = []
            if gold_database_source == "dataset":
                # prepare database based on data type
                if data_type == "gazetteers":
                    items = defaultdict(dict)
                    for part in ["train", "dev", "test"]:
                        with open(files[part], "r", encoding="utf-8") as file:
                            docs = json.load(file)
                        for doc in docs:
                            for entity in doc["entities"]:
                                ne = " ".join(doc["tokens"]
                                              [entity["start"]:entity["end"]])
                                key = entity["type"] + "_" + ne
                                if "doc_id" not in items[key]:
                                    items[key]["doc_id"] = []
                                if doc["doc_id"] not in items[key]["doc_id"]:
                                    items[key]["doc_id"].append(doc["doc_id"])
                                if "dataset" not in items[key]:
                                    items[key]["dataset"] = []
                                if part not in items[key]["dataset"]:
                                    items[key]["dataset"].append(part)
                                items[key]["type"] = entity["type"]
                                items[key]["content"] = ne
                    documents = [
                        Document(content=doc["content"],
                                 meta={
                                     "doc_id": doc["doc_id"],
                                     "dataset": doc["dataset"],
                                     "type": doc["type"],
                                     "data_type": data_type
                                 }) for doc in items.values()
                    ]

                elif data_type == "sentences":
                    for part in ["train", "dev", "test"]:
                        docs = json.load(files[part])
                        for doc in docs:
                            documents.append(
                                Document(content=" ".join(doc["tokens"]),
                                         meta={
                                             "entities": doc["entities"],
                                             "data_type": data_type,
                                             "doc_id": [doc["doc_id"]]
                                         }))
                else:
                    raise Exception(
                        "Argument error: data_type - must be: gazetteers | sentences, but is: "
                        + data_type)
            gold_document_store.write_documents(documents)

        if search_algorithm != "bm25":
            if gold_document_store.get_document_count(
            ) > gold_document_store.get_embedding_count():
                gold_document_store.update_embeddings(
                    gold_search.get_node("ANNRetriever"),  # type: ignore 
                    update_existing_embeddings=False)

        # 3. Augment train + dev (+ test if validate_on_test) set with gold_database
        # 3.1 search for similar documents; filter out exact matches
        files["tokenized_gold_train"] = tokenize_database_json(
            tokenizer,
            files["train"],
            files["types"],
            gold_search,
            use_labels,
            use_mentions,
            "tokenized_gold_train",
            filters={"dataset": ["train"]},
            prepend_search_results=prepend_search_results)
        files["tokenized_gold_dev"] = tokenize_database_json(
            tokenizer,
            files["dev"],
            files["types"],
            gold_search,
            use_labels,
            use_mentions,
            "tokenized_gold_dev",
            filters={"dataset": ["train", "dev"]},
            prepend_search_results=prepend_search_results)
        files["tokenized_gold_test"] = tokenize_database_json(
            tokenizer,
            files["test"],
            files["types"],
            gold_search,
            use_labels,
            use_mentions,
            "tokenized_gold_test",
            filters={"dataset": ["train", "dev", "test"]},
            prepend_search_results=prepend_search_results)

        # 4. Train baseline with gold database
        trained_gold = False
        while not trained_gold:
            try:
                (gold_train_result, gold_dev_result, gold_test_result, _, _,
                 _) = run_experiment(
                     name=name + "_gold",
                     tokenized_train_data_file=files["tokenized_gold_train"],
                     tokenized_dev_data_file=files["tokenized_gold_dev"],
                     tokenized_test_data_file=files["tokenized_gold_test"],
                     type_data_file=files["types"],
                     logger_dir_path=run_dir_path,
                     config=model_config,
                     validate_on_test=validate_on_test)
                trained_gold = True
                # 4.1 report metrics
                model_metrics[run_id]["gold"] = {}
                model_metrics[run_id]["gold"]["train_f1"] = gold_train_result[
                    0]["val_f1_epoch"]  # type: ignore
                model_metrics[run_id]["gold"]["dev_f1"] = gold_dev_result[0][
                    "val_f1_epoch"]  # type: ignore
                if gold_test_result is not None:
                    model_metrics[run_id]["gold"][
                        "test_f1"] = gold_test_result[0][
                            "test_f1_epoch"]  # type: ignore
            except Exception:
                model_config["gradient_accumulation_steps"] += 1
                model_config[
                    "batch_size"] = model_config["batch_size"] // model_config[
                        "gradient_accumulation_steps"]


# 5. Prepare errorneous database
# 5.1 create new database: error_document_store = name + "error" + gold_database_source + seed
# 5.1.1 copy items from gold_document_store
# 5.1.2 store and filter out random database entries until error_database_size is reached
# 5.1.3 add false positives to database
# 5.1.4 compute embeddings if search_algorithm != bm25

# 6. Augment train + dev (+ test if validate_on_test) set with error_database
# 6.1 search for similar documents; filter out exact matches

# 7. Train baseline with errorneous database
# 7.1 report metrics

# 8. Validate on error corrections
# 8.1 Prepare mLOWNER testset database (for extend)
# 8.1.1 Filter out exact matching items from train + dev + test set of the dataset
# 8.2 Delete
# 8.2.1 Copy erroneous database into delete_document_store = name + "error_delete" + gold_database_source + seed
# 8.2.2 label false positives with "O" -> remove from items "entities" list
# 8.2.3 Augment dev / test set with this version of errorneous database
# 8.2.4 Validate model in 7. on augmented dev / test set
# 8.2.5 report metrics
# 8.3 Update
# 8.3.1 Copy erroneous database into delete_document_store = name + "error_update" + gold_database_source + seed
# 8.3.2 label false positives with correct labels -> item's "entities" list = labels from dataset item
# 8.3.3 Augment dev / test set with this version of errorneous database
# 8.3.4 Validate model in 7. on augmented dev / test set
# 8.3.5 report metrics
# 8.4 Add
# 8.4.1 Copy erroneous database into delete_document_store = name + "error_add" + gold_database_source + seed
# 8.4.2 Add filtered out items from 5.1.2
# 8.4.3 Augment dev / test set with this version of errorneous database
# 8.4.4 Validate model in 7. on augmented dev / test set
# 8.4.5 report metrics
# 8.5 Extend
# 8.5.1 Copy erroneous database into delete_document_store = name + "error_extend" + gold_database_source + seed
# 8.5.2 Get + index similar items for train / + dev set from mLOWNER testset database in 8.1
# 8.5.3 Augment dev / test set with this version of errorneous database
# 8.5.4 Validate model in 7. on augmented dev / test set
# 8.5.5 report metrics
# 8.6 Delete + Add
# 8.7 Delete + Extend
# 8.8 Update + Add
# 8.9 Update + Extend
# 8.10 Delete + Add + Extend
# 8.11 Update + Add + Extend

# 9. Report database metrics
# 9.1 Overlap of dataset + databases
# 10. Save best run based on overall F1 score
    with open(os.path.join(eval_dir_path, "model_metrics.json"),
              "w",
              encoding="utf-8") as file:
        json.dump(model_metrics, file)
    with open(os.path.join(eval_dir_path, "database_metrics.json"),
              "w",
              encoding="utf-8") as file:
        json.dump(database_metrics, file)
