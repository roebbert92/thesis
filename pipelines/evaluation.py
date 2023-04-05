
def run_evaluation(
        name: str,
        dataset_files: dict,
        search_algorithm: str, # bm25, ann, ann+ranking
        search_topk: int, # > 0
        gold_database_source: str, # lowner, dataset
        error_database_size: float, # 0...1 relative to gold_database_size
        data_type: str, # sentences, gazetteers
        input_processing: str, # fusion-in-decoder, few-shot
        use_labels: bool, # :<label>
        use_mentions: bool, # <m></m>
        sampling: str, # topk, nucleus
        sampling_value: float, # > 0
        beam_size: int, # 1 = greedy search; > 0
        validate_on_test: bool, # False for validating on devset, True for validating on testset
        num_runs: int # > 1; how many runs
):
    # 0. loop over num_runs
    # 0.1 lightning seed everything with new random seed (clear environment variable PL_GLOBAL_SEED)

    # 1. Train baseline
    # 1.1 collect false positives from train + dev
    # 1.2 report metrics

    # 2. Prepare gold database
        # gold_document_store = name + "gold" + gold_database_source + seed
    # 2.1 Load into search engine if gold_document_store does not exist

    # 3. Augment train + dev (+ test if validate_on_test) set with gold_database
    # 3.1 search for similar documents; filter out exact matches

    # 4. Train baseline with gold database
    # 4.1 report metrics

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