from argparse import ArgumentError
from collections import defaultdict
import copy
import json
import sys
import os

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from tqdm import tqdm
import lightning.fabric.utilities.seed as pl_seed
import shutil
import numpy as np
from haystack import Pipeline, Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever, SentenceTransformersRanker, BM25Retriever

from data_preprocessing.tokenize import tokenize_json, tokenize_database_json
from models.asp_t5 import get_tokenizer
from pipelines.asp_training import run_experiment


def setup_database(database_name: str, search_algorithm: str,
                   search_topk: int):
    gold_document_store = ElasticsearchDocumentStore(index=database_name)
    gold_search = Pipeline()
    if search_algorithm == "bm25":
        bm25_retriever = BM25Retriever(gold_document_store, top_k=search_topk)
        gold_search.add_node(component=bm25_retriever,
                             name="BM25Retriever",
                             inputs=["Query"])
    elif search_algorithm.startswith("ann"):
        ann_retriever = EmbeddingRetriever(
            document_store=gold_document_store,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            model_format="sentence_transformers",
            top_k=search_topk *
            2 if search_algorithm.endswith("ranking") else search_topk)
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

    return gold_document_store, gold_search


def augment_dataset(name,
                    tokenizer,
                    files,
                    database,
                    use_labels,
                    use_mentions,
                    prepend_search_results,
                    filtered_document_ids={}):
    train_name = "tokenized_" + name + "_train"
    files[train_name] = tokenize_database_json(
        tokenizer,
        files["train"],
        files["types"],
        database,
        use_labels,
        use_mentions,
        train_name,
        filters={"$or": [{
            "dataset": ["train"]
        }]},
        filtered_document_ids=filtered_document_ids["train"]
        if "train" in filtered_document_ids else [],
        prepend_search_results=prepend_search_results)
    dev_name = "tokenized_" + name + "_dev"
    files[dev_name] = tokenize_database_json(
        tokenizer,
        files["dev"],
        files["types"],
        database,
        use_labels,
        use_mentions,
        dev_name,
        filters={"$or": [{
            "dataset": ["train"]
        }, {
            "dataset": ["dev"]
        }]},
        filtered_document_ids=filtered_document_ids["dev"]
        if "dev" in filtered_document_ids else [],
        prepend_search_results=prepend_search_results)
    test_name = "tokenized_" + name + "_test"
    files[test_name] = tokenize_database_json(
        tokenizer,
        files["test"],
        files["types"],
        database,
        use_labels,
        use_mentions,
        test_name,
        filters={
            "$or": [{
                "dataset": ["train"]
            }, {
                "dataset": ["dev"]
            }, {
                "dataset": ["test"]
            }]
        },
        filtered_document_ids=filtered_document_ids["test"]
        if "test" in filtered_document_ids else [],
        prepend_search_results=prepend_search_results)


def report_metrics(model_metrics, run_id, name, train_result, dev_result,
                   test_result):
    model_metrics[run_id][name] = {}
    model_metrics[run_id][name]["train_f1"] = train_result[0][
        "val_f1_epoch"]  # type: ignore
    model_metrics[run_id][name]["dev_f1"] = dev_result[0][
        "val_f1_epoch"]  # type: ignore
    if test_result is not None:
        model_metrics[run_id][name]["test_f1"] = test_result[0][
            "test_f1_epoch"]  # type: ignore


def add_false_entities(doc_entities, false_entities):
    overlapped_entities = []
    new_entities = []
    for false_entity in false_entities:
        # check if false_entities overlap -> take longer one:
        added_false_entity = False
        for idx, new_entity in enumerate(list(new_entities)):
            if false_entity["start"] <= new_entity["start"] <= false_entity[
                    "end"] or false_entity["start"] <= new_entity[
                        "end"] <= false_entity["end"]:
                len_new_entity = new_entity["end"] - new_entity["start"]
                len_false_entity = false_entity["end"] - false_entity["start"]
                if len_new_entity < len_false_entity:
                    new_entities[idx] = false_entity
                    added_false_entity = True
        if not added_false_entity:
            new_entities.append(false_entity)

    for false_entity in new_entities:
        # check if entities overlap
        for doc_entity in doc_entities:
            if false_entity["start"] <= doc_entity["start"] <= false_entity[
                    "end"] or false_entity["start"] <= doc_entity[
                        "end"] <= false_entity["end"]:
                # entity overlap
                overlapped_entities.append(doc_entity)

    for doc_entity in doc_entities:
        if doc_entity not in overlapped_entities:
            new_entities.append(doc_entity)
    return new_entities


def get_or_filter_from_list(key_name, values):
    return {"$or": [{key_name: value} for value in values]}


def transform_false_positives(id_to_label, false_positives):
    result = {}
    for key, values in false_positives.items():
        result[key] = [{
            "start": start,
            "end": end,
            "type": id_to_label[t]
        } for (start, end, t) in values]
    return result


def get_documents_from_gazetteers(docs):
    items = defaultdict(dict)
    for doc in docs:
        for entity in doc["entities"]:
            ne = " ".join(doc["tokens"][entity["start"]:entity["end"]])
            key = entity["type"] + "_" + ne
            if "doc_id" not in items[key]:
                items[key]["doc_id"] = []
            if doc["doc_id"] not in items[key]["doc_id"]:
                items[key]["doc_id"].append(doc["doc_id"])
            if "dataset" not in items[key]:
                items[key]["dataset"] = []
            dataset_part = doc["doc_id"].split("_")[1]
            if dataset_part not in items[key]["dataset"]:
                items[key]["dataset"].append(dataset_part)
            items[key]["type"] = entity["type"]
            items[key]["content"] = ne
    return [
        Document(content=doc["content"],
                 meta={
                     "doc_id": doc["doc_id"],
                     "dataset": doc["dataset"],
                     "type": doc["type"],
                     "data_type": "gazetteers"
                 }) for doc in items.values()
    ]


def get_documents_from_sentences(docs):
    documents = []
    for doc in docs:
        dataset_part = doc["doc_id"].split("_")[1]
        documents.append(
            Document(content=" ".join(doc["tokens"]),
                     meta={
                         "entities": doc["entities"],
                         "data_type": "sentences",
                         "doc_id": [doc["doc_id"]],
                         "dataset": [dataset_part],
                     }))
    return documents


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

        model_config["seed"] = seed
        model_config["beam_size"] = beam_size
        model_config["input_processing"] = input_processing
        model_config["sampling"] = sampling
        model_config["sampling_value"] = sampling_value

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
        (baseline_train_result, baseline_dev_result, baseline_test_result,
         baseline_train_false_positives, baseline_dev_false_positives,
         baseline_test_false_positives) = run_experiment(
             name=name,
             tokenized_train_data_file=files["tokenized_train"],
             tokenized_dev_data_file=files["tokenized_dev"],
             tokenized_test_data_file=files["tokenized_train"],
             type_data_file=files["types"],
             logger_dir_path=run_dir_path,
             config=model_config,
             validate_on_test=validate_on_test)

        with open(files["types"], encoding="utf-8") as file:
            labels = json.load(file)['entities']
        id_to_label = {id: label for id, label in enumerate(labels)}

        false_positives = {
            "train":
            transform_false_positives(id_to_label,
                                      baseline_train_false_positives),
            "dev":
            transform_false_positives(
                id_to_label, baseline_train_false_positives
                | baseline_dev_false_positives),
            "test":
            transform_false_positives(
                id_to_label, baseline_train_false_positives
                | baseline_dev_false_positives
                | baseline_test_false_positives
                if baseline_test_false_positives is not None else
                baseline_train_false_positives
                | baseline_dev_false_positives)
        }
        # 1.2.2 report metrics
        report_metrics(model_metrics, run_id, "baseline",
                       baseline_train_result, baseline_dev_result,
                       baseline_test_result)

        # 2. Prepare gold database
        # gold_document_store = name + "gold" + gold_database_source + seed
        gold_document_store, gold_search = setup_database(
            name + "_gold_" + gold_database_source + "_" + str(seed),
            search_algorithm, search_topk)

        # 2.1 Load dataset jsons
        # 2.1 Load into search engine if gold_document_store does not exist
        if gold_document_store.get_document_count() == 0:
            documents = []
            if gold_database_source == "dataset":
                # prepare database based on data type
                docs = []
                for part in ["train", "dev", "test"]:
                    with open(files[part], "r", encoding="utf-8") as file:
                        docs.extend(json.load(file))
                if data_type == "gazetteers":
                    documents = get_documents_from_gazetteers(docs)
                elif data_type == "sentences":
                    documents = get_documents_from_sentences(docs)
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
        augment_dataset("gold", tokenizer, files, gold_search, use_labels,
                        use_mentions, prepend_search_results)

        # 4. Train baseline with gold database
        trained_gold = False
        gold_config = copy.deepcopy(model_config)
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
                     config=gold_config,
                     validate_on_test=validate_on_test)
                trained_gold = True
                # 4.1 report metrics
                report_metrics(model_metrics, run_id, "gold",
                               gold_train_result, gold_dev_result,
                               gold_test_result)
            except Exception:
                gold_config["gradient_accumulation_steps"] += 1
                gold_config["batch_size"] = gold_config[
                    "batch_size"] // gold_config["gradient_accumulation_steps"]

        # 5. Prepare errorneous database
        # 5.1 create new database: error_document_store = name + "error" + gold_database_source + seed
        error_document_store, error_search = setup_database(
            name + "_error_" + gold_database_source + "_" + str(seed),
            search_algorithm, search_topk)
        # 5.1.1 copy items from gold_document_store
        filters = {
            "train": ["train"],
            "dev": ["train", "dev"],
            "test": ["train", "dev", "test"]
        }

        # Get original doc from dataset json
        original_docs = dict()
        for file_name in [files[file] for file in filters["test"]]:
            with open(file_name, "r", encoding="utf-8") as file:
                for item in json.load(file):
                    original_docs[item["doc_id"]] = item
        # 5.1.2 add false positives to database
        for doc_id, false_entities in false_positives["test"].items():
            # sentences
            original_docs[doc_id]["entities"] = add_false_entities(
                original_docs[doc_id]["entities"], false_entities)

        documents = []
        if data_type == "gazetteers":
            documents = get_documents_from_gazetteers(original_docs.values())
        elif data_type == "sentences":
            documents = get_documents_from_sentences(original_docs.values())

        error_document_store.write_documents(documents)
        # 5.1.3 compute embeddings if search_algorithm != bm25
        if search_algorithm != "bm25":
            if error_document_store.get_document_count(
            ) > error_document_store.get_embedding_count():
                error_document_store.update_embeddings(
                    error_search.get_node("ANNRetriever"),  # type: ignore 
                    update_existing_embeddings=False)

        # 5.1.4 store and filter out random database entries until error_database_size is reached
        error_document_id_filter = defaultdict(list)
        rng = np.random.default_rng(seed=seed)
        for part in filters:
            current_filter = get_or_filter_from_list("dataset", filters[part])
            current_filter["$not"] = {  # type: ignore
                "doc_id": [doc_id for doc_id in false_positives[part]]
            }
            doc_count = error_document_store.get_document_count(
                current_filter)  # type: ignore
            filter_mask = rng.choice(doc_count,
                                     int(doc_count * error_database_size),
                                     replace=False).tolist()
            docs = error_document_store.get_all_documents(
                filters=current_filter)  # type: ignore
            for mask, doc in enumerate(docs):
                if mask in filter_mask and doc.id not in error_document_id_filter[
                        part]:
                    error_document_id_filter[part].append(doc.id)

        # 6. Augment train + dev (+ test if validate_on_test) set with error_database
        # 6.1 search for similar documents; filter out exact matches
        augment_dataset("error", tokenizer, files, error_search, use_labels,
                        use_mentions, prepend_search_results,
                        error_document_id_filter)

        #  7. Train baseline with errorneous database
        trained_error = False
        error_config = copy.deepcopy(model_config)
        while not trained_error:
            try:
                (error_train_result, error_dev_result, error_test_result, _, _,
                 _) = run_experiment(
                     name=name + "_error",
                     tokenized_train_data_file=files["tokenized_error_train"],
                     tokenized_dev_data_file=files["tokenized_error_dev"],
                     tokenized_test_data_file=files["tokenized_error_test"],
                     type_data_file=files["types"],
                     logger_dir_path=run_dir_path,
                     config=error_config,
                     validate_on_test=validate_on_test)
                trained_error = True
                # 7.1 report metrics
                report_metrics(model_metrics, run_id, "error",
                               error_train_result, error_dev_result,
                               error_test_result)
            except Exception:
                error_config["gradient_accumulation_steps"] += 1
                error_config[
                    "batch_size"] = error_config["batch_size"] // error_config[
                        "gradient_accumulation_steps"]


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
