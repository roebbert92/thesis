import numpy as np
import seaborn as sns
from locale import strcoll
from functools import cmp_to_key
import pandas as pd

def compare_dataset_names(a, b):
    scores = {"train": 0, "dev": 1, "test": 2, "gaz": 3}
    a1, a2 = a.split("_")
    b1, b2 = b.split("_")
    dataset_eq = strcoll(a1, b1)
    if dataset_eq != 0:
        return dataset_eq
    if a2 == b2:
        return 0
    return scores[a2] - scores[b2]

def visualize_confusion_matrix(data: pd.DataFrame, metric_column: str):
    conf_matrix = data.pivot_table(metric_column,
                                   index="first",
                                   columns="second",
                                   aggfunc=np.mean)
    datasets = conf_matrix.columns.to_list()
    cmp_dataset_names = cmp_to_key(compare_dataset_names)
    datasets.sort(key=cmp_dataset_names)
    conf_matrix = conf_matrix[datasets].sort_index(
        level="first", key=lambda x: pd.Series([datasets.index(y) for y in x]))
    sns_plot = sns.heatmap(conf_matrix,
                           annot=True,
                           cmap=sns.color_palette("Blues", as_cmap=True))
    return sns_plot.get_figure()