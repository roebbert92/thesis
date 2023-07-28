# Adapting NER taggers to emerging entities with gazetteers

Thesis by Robin Löbbert, 31.07.2023

## Outline

1. Configuration
   - configs
2. Search
   - search
3. Models
   - models
4. Data Preprocessing
   - data
   - data_preparation
   - data_metrics
   - data_preprocessing
5. Hyperparameter tuning
   - hyperparameter_tuning
6. Experiments
   1. experiments > 01_performance
   2. experiments > 02_content
      - data_augmentation
   3. experiments > 03_adaptation_emerging_entities
7. Evaluation
   - evaluation

frameworks is just a storage directory for code of T5-ASP.

## Reproduce results

Reproducing the results for T5-ASP, just run run.py in the respective experiment folder.
To reproduce results for FLAIR, run models > flair_roberta.py.
To reproduce results for DictMatch, run models > dict_match.py.
To reproduce results for SearchMatch, run models > search_match.py.

## Evaluate results

Run the respective Juypter notebooks after running the experiments.
