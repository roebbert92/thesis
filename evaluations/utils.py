from typing import Dict, List, Optional
import pandas as pd
from itertools import product
from pandas.io.formats.style import Styler
import matplotlib as plt

plt.rcParams.update({
    'mathtext.default': 'regular',
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 14
})

MODEL_ORDER = {
    key: idx
    for idx, key in enumerate([
        "flair_roberta", "t5_asp", "dict_match", "dict_match_gaz",
        "dict_match_sent", "dict_match_lownergaz", "dict_match_gaz_sent",
        "dict_match_lownergaz_sent", "search_match", "search_match_gaz",
        "search_match_sent", "search_match_lownergaz", "search_match_gaz_sent",
        "search_match_lownergaz_sent", "search_match_overall", "t5_asp_gaz",
        "t5_asp_sent", "t5_asp_lownergaz", "t5_asp_gaz_sent",
        "t5_asp_lownergaz_sent", "t5_asp_overall"
    ])
}
LATEX_MODEL_NAMES = {
    "flair_roberta": "FLAIR\\textsubscript{XLM-RoBERTa-Large}",
    "t5_asp": "T5-ASP",
    "dict_match": "DictMatch",
    "dict_match_gaz": "DictMatch\\textsubscript{Gaz}",
    "dict_match_sent": "DictMatch\\textsubscript{Sent}",
    "dict_match_lownergaz": "DictMatch\\textsubscript{LownerGaz}",
    "dict_match_lownergaz_sent": "DictMatch\\textsubscript{LownerGaz+Sent}",
    "dict_match_gaz_sent": "DictMatch\\textsubscript{Gaz+Sent}",
    "search_match": "SearchMatch",
    "search_match_gaz": "SearchMatch\\textsubscript{Gaz}",
    "search_match_sent": "SearchMatch\\textsubscript{Sent}",
    "search_match_lownergaz": "SearchMatch\\textsubscript{LownerGaz}",
    "search_match_lownergaz_sent":
    "SearchMatch\\textsubscript{LownerGaz+Sent}",
    "search_match_gaz_sent": "SearchMatch\\textsubscript{Gaz+Sent}",
    "t5_asp_gaz": "T5-ASP\\textsubscript{Gaz}",
    "t5_asp_sent": "T5-ASP\\textsubscript{Sent}",
    "t5_asp_lownergaz": "T5-ASP\\textsubscript{LownerGaz}",
    "t5_asp_gaz_sent": "T5-ASP\\textsubscript{Gaz+Sent}",
    "t5_asp_lownergaz_sent": "T5-ASP\\textsubscript{LownerGaz+Sent}",
    "search_match_overall": "SearchMatch\\textsubscript{Overall}",
    "t5_asp_overall": "T5-ASP\\textsubscript{Overall}"
}

DATASET_ORDER = {
    key: idx
    for idx, key in enumerate([
        'lowner_train', 'lowner_dev', 'lowner_test', 'lowner_gazetteer',
        'multiconer_test', 'wnut_train', 'wnut_dev', 'wnut_test'
    ])
}

LATEX_DATASET_NAMES = {
    'lowner_train': "LOWNER\\textsubscript{train}",
    'lowner_dev': "LOWNER\\textsubscript{dev}",
    'lowner_test': "LOWNER\\textsubscript{test}",
    'lowner_gazetteer': "LOWNER\\textsubscript{gazetteer}",
    'multiconer_test': "MultiCoNER\\textsubscript{test}",
    'wnut_train': "WNUT-17\\textsubscript{train}",
    'wnut_dev': "WNUT-17\\textsubscript{dev}",
    'wnut_test': "WNUT-17\\textsubscript{test}"
}

PLOT_DATASET_NAMES = {
    'lowner_train': r"LOWNER$_{train}$",
    'lowner_dev': r"LOWNER$_{dev}$",
    'lowner_test': r"LOWNER$_{test}$",
    'lowner_gazetteer': r"LOWNER$_{gazetteer}$",
    'multiconer_test': r"MultiCoNER$_{test}$",
    'wnut_train': r"WNUT-17$_{train}$",
    'wnut_dev': r"WNUT-17$_{dev}$",
    'wnut_test': r"WNUT-17$_{test}$"
}

METRIC_ORDER = {
    key: idx
    for idx, key in enumerate([
        "eecr_labeled_data", "eecr_labeled_data_per_sample",
        "eecr_search_results", "max", "score", "search score", "error_type1",
        "error_type2", "error_type3", "error_type4", "error_type5",
        "error rate", "tp", "fn", "fp", "precision", "recall", "f1", "targets"
    ])
}

LATEX_METRIC_NAMES = {
    "error_type1": "Error Type-1",
    "error_type2": "Error Type-2",
    "error_type3": "Error Type-3",
    "error_type4": "Error Type-4",
    "error_type5": "Error Type-5",
    "fn": "False negatives",
    "fp": "False positives",
    "error rate": "Error rate",
    "tp": "True positives",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Micro-F1",
    "targets": "Input entity count",
    "eecr_labeled_data": "EECR labeled data",
    "eecr_labeled_data_per_sample": "EECR labeled data per sample",
    "eecr_search_results": "EECR search results",
    "max": "Top-1 Cosine Similarity",
    "score": "Search score",
    "search score": "Search score",
    "total entities": "Retrieved entities"
}

PLOT_MODEL_NAMES = {
    "flair_roberta": r"FLAIR",
    "t5_asp": r"T5-ASP",
    "dict_match": r"DictMatch",
    "dict_match_gaz": r"DictMatch$_{Gaz}$",
    "dict_match_sent": r"DictMatch$_{Sent}$",
    "dict_match_lownergaz": r"DictMatch$_{LownerGaz}$",
    "dict_match_lownergaz_sent": r"DictMatch$_{LownerGaz+Sent}$",
    "dict_match_gaz_sent": r"DictMatch$_{Gaz+Sent}$",
    "search_match": r"SearchMatch",
    "search_match_gaz": r"SearchMatch$_{Gaz}$",
    "search_match_sent": r"SearchMatch$_{Sent}$",
    "search_match_lownergaz": r"SearchMatch$_{LownerGaz}$",
    "search_match_lownergaz_sent": r"SearchMatch$_{LownerGaz+Sent}$",
    "search_match_gaz_sent": r"SearchMatch$_{Gaz+Sent}$",
    "t5_asp_gaz": r"T5-ASP$_{Gaz}$",
    "t5_asp_sent": r"T5-ASP$_{Sent}$",
    "t5_asp_lownergaz": r"T5-ASP$_{LownerGaz}$",
    "t5_asp_gaz_sent": r"T5-ASP$_{Gaz+Sent}$",
    "t5_asp_lownergaz_sent": r"T5-ASP$_{LownerGaz+Sent}$"
}

PLOT_SEARCH_NAMES = {
    "t5_asp_gaz": "Gaz",
    "t5_asp_sent": "Sent",
    "t5_asp_lownergaz": "LownerGaz",
    "t5_asp_lownergaz_gaz": "LownerGaz+Gaz",
    "t5_asp_gaz_sent": "Gaz+Sent",
    "t5_asp_lownergaz_sent": "LownerGaz+Sent"
}

ENTITY_NAMES = {
    "person": "Person",
    "location": "Location",
    "group": "Group",
    "corporation": "Corporation",
    "creative-work": "Creative Work",
    "product": "Product"
}


def merge_mean_std(dataframe: pd.DataFrame,
                   mean_column,
                   std_column,
                   precision=2) -> pd.Series:
    return dataframe[mean_column].map(
        lambda x: '{:,.{prec}f}'.format(x, prec=precision)) + " (" + (round(
            dataframe[std_column] *
            10**precision)).astype(int).astype(str) + ")"


def highlight_correlations(styler: Styler):
    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    cm_blue = plt.colormaps["Blues"].resampled(len(ranges))
    for i, (left, right) in enumerate(ranges):
        color = plt.colors.rgb2hex(cm_blue(i))[1:].upper()
        if i > 0:
            is_last = i >= len(ranges) - 2
            styler.highlight_between(
                left=left,
                right=right,
                inclusive="left",
                props="cellcolor:[HTML]{" + color +
                "};color:{white}" if is_last else "cellcolor:[HTML]{" + color +
                "};")
            styler.highlight_between(
                left=-right,
                right=-left,
                inclusive="right",
                props="cellcolor:[HTML]{" + color +
                "};color:{white}" if is_last else "cellcolor:[HTML]{" + color +
                "};")
    return styler


def get_dataset_stats_totals_correct_latex_format(df: pd.DataFrame):
    major_df = df[[
        "dataset", "total samples", "sample len avg", "sample len std",
        "total entities", "entity count avg", "entity count std"
    ]]
    round_last_digits = 2
    major_df["Dataset"] = major_df["dataset"].apply(
        lambda x: LATEX_DATASET_NAMES[x])
    major_df["Total samples"] = major_df["total samples"]
    major_df["Sample length"] = major_df["sample len avg"].map(
        lambda x: '{:,.{prec}f}'.format(x, prec=round_last_digits)) + " (" + (
            round(major_df["sample len std"] *
                  10**round_last_digits)).astype(int).astype(str) + ")"
    major_df["Total entities"] = major_df["total entities"]
    major_df["Entity count"] = major_df["entity count avg"].map(
        lambda x: '{:,.{prec}f}'.format(x, prec=round_last_digits)) + " (" + (
            round(major_df["entity count std"] *
                  10**round_last_digits)).astype(int).astype(str) + ")"

    major_df.set_index("Dataset", inplace=True)
    return major_df[[
        "Total samples", "Sample length", "Total entities", "Entity count"
    ]].style.to_latex(
        siunitx=True,
        hrules=True,
        multirow_align="t",
        column_format=
        "@{}llS[table-format = 7]S[table-format = 2.2 (2)]S[table-format = 7]S[table-format = 2.2 (2)]@{} "
    )


def get_gazetteer_stats_totals_correct_latex_format(df: pd.DataFrame):
    major_df = df[[
        "gazetteer", "total samples", "sample len avg", "sample len std",
        "total entities", "entity count avg", "entity count std",
        "distinct entities"
    ]]
    round_last_digits = 2
    major_df["Gazetteer"] = major_df["gazetteer"].apply(
        lambda x: PLOT_SEARCH_NAMES[x])
    major_df["Total samples"] = major_df["total samples"].copy(deep=True)
    major_df["Sample length"] = major_df["sample len avg"].map(
        lambda x: '{:,.{prec}f}'.format(x, prec=round_last_digits)) + " (" + (
            round(major_df["sample len std"] *
                  10**round_last_digits)).astype(int).astype(str) + ")"
    major_df["Total entities"] = major_df["total entities"].copy(deep=True)
    major_df["Distinct entities"] = major_df["distinct entities"].copy(
        deep=True)
    major_df["Entity count"] = major_df["entity count avg"].map(
        lambda x: '{:,.{prec}f}'.format(x, prec=round_last_digits)) + " (" + (
            round(major_df["entity count std"] *
                  10**round_last_digits)).astype(int).astype(str) + ")"

    major_df.set_index("Gazetteer", inplace=True)
    return major_df[[
        "Total samples", "Sample length", "Total entities", "Entity count",
        "Distinct entities"
    ]].style.to_latex(
        siunitx=True,
        hrules=True,
        multirow_align="t",
        column_format=
        "@{}llS[table-format = 7]S[table-format = 2.2 (2)]S[table-format = 7]S[table-format = 2.2 (2)]S[table-format = 7]@{} "
    )


def get_gazetteer_stats_entity_types_correct_latex_format(df: pd.DataFrame):
    type_df = df[["gazetteer", *ENTITY_NAMES.keys()]]
    type_df["Gazetteer"] = type_df["gazetteer"].apply(
        lambda x: PLOT_SEARCH_NAMES[x])
    for entity_type, entity_name in ENTITY_NAMES.items():
        type_df[entity_name] = type_df[entity_type]

    type_df.set_index("Gazetteer", inplace=True)

    return type_df[[*ENTITY_NAMES.values()
                    ]].style.to_latex(siunitx=True,
                                      hrules=True,
                                      multirow_align="t",
                                      column_format="@{}lSSSSSS@{}")


def get_dataset_stats_entity_types_correct_latex_format(df: pd.DataFrame):
    type_df = df[["dataset", *ENTITY_NAMES.keys()]]
    type_df["Dataset"] = type_df["dataset"].apply(
        lambda x: LATEX_DATASET_NAMES[x])
    for entity_type, entity_name in ENTITY_NAMES.items():
        type_df[entity_name] = type_df[entity_type].copy(deep=True)

    type_df.set_index("Dataset", inplace=True)

    return type_df[[*ENTITY_NAMES.values()
                    ]].style.to_latex(siunitx=True,
                                      hrules=True,
                                      multirow_align="t",
                                      column_format="@{}lSSSSSS@{}")


def get_correlations_correct_latex_format(df: pd.DataFrame,
                                          model_names: Optional[dict] = None):
    def index_order(x: pd.Index):
        if x.name == "perf_metric":
            return x.map(METRIC_ORDER)
        elif x.name == "model":
            return x.map(MODEL_ORDER)
        return x

    def column_order(x: pd.Index):
        if x.name == "data_metric":
            return x.map(METRIC_ORDER)
        return x

    if model_names is None:
        model_names = LATEX_MODEL_NAMES

    df.sort_index(axis=0, inplace=True, key=index_order)
    df.sort_index(axis=1, inplace=True, key=column_order)
    df.index = df.index.map(lambda x:
                            (LATEX_METRIC_NAMES[x[0]], model_names[x[1]]))
    df.columns = df.columns.map(LATEX_METRIC_NAMES)
    df.index.rename(["Performance Metric", "Model type"], inplace=True)
    df.columns.rename("Input Data Metric", inplace=True)

    return df.style.pipe(highlight_correlations).format('{:.4f}').to_latex(
        siunitx=True,
        hrules=True,
        multirow_align="t",
        column_format=
        "@{} ll*4{S[table-format = 1.4, group-minimum-digits=5]}@{} ")


def get_correct_latex_format(df: pd.DataFrame,
                             columns: List[str],
                             column_names: Dict[str, str],
                             round_last_digits=2):
    # sort by model name
    sorted_df = df[["model", *columns]].sort_values(
        "model", key=lambda x: x.apply(lambda y: MODEL_ORDER.get(y, 1000)))

    # rename models to latex names
    sorted_df["Models"] = sorted_df["model"].apply(
        lambda x: LATEX_MODEL_NAMES[x])
    # combine mean + std in one column
    column_depth = sorted_df.columns.nlevels
    if column_depth == 1:
        for column, new_column in column_names.items():
            sorted_df[new_column] = sorted_df[column].map(
                lambda x: '{:,.{prec}f}'.format(x, prec=round_last_digits))
        result_df = sorted_df[["Models", *list(column_names.values())
                               ]].set_index("Models")
    elif column_depth == 2:
        for column, new_column in column_names.items():
            sorted_df[new_column] = sorted_df[column]['mean'].map(
                lambda x: '{:,.{prec}f}'.format(x, prec=round_last_digits)
            ) + " (" + (round(sorted_df[column]['std'] * 10**
                              round_last_digits)).astype(int).astype(str) + ")"
        result_df = sorted_df[["Models", *list(column_names.values())
                               ]].set_index("Models")
        result_df.columns = ["".join(col) for col in result_df.columns.values]
    elif column_depth == 3:
        # iterate over column names - entities
        for level_0_col, ent_name in ENTITY_NAMES.items():
            for column, new_column in column_names.items():
                sorted_df[ent_name, new_column,
                          ""] = sorted_df[level_0_col][column]['mean'].map(
                              lambda x: '{:,.{prec}f}'.format(
                                  x, prec=round_last_digits)) + " (" + (round(
                                      sorted_df[level_0_col][column]['std'] *
                                      100)).astype(int).astype(str) + ")"
        result_df = sorted_df[[
            ("Models", "", ""),
            *list(product(ENTITY_NAMES.values(), column_names.values(), [""]))
        ]].set_index("Models")
        if len(column_names) == 1:
            result_df.columns = [col[0] for col in result_df.columns.values]
        else:
            result_df.columns = [
                tuple(c for c in col[:-1]) for col in result_df.columns.values
            ]
    else:
        result_df = pd.DataFrame()

    return result_df.to_latex(bold_rows=True, escape=False)