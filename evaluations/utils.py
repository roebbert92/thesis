from typing import Dict, List
import pandas as pd
from itertools import product


def get_correct_latex_format(df: pd.DataFrame, columns: List[str],
                             column_names: Dict[str, str]):
    # sort by model name
    model_order = {
        key: idx
        for idx, key in enumerate([
            "flair_roberta", "t5_asp", "dict_match_gaz", "dict_match_sent",
            "dict_match_lownergaz", "dict_match_gaz_sent",
            "dict_match_lownergaz_sent", "t5_asp_gaz", "t5_asp_sent",
            "t5_asp_lownergaz", "t5_asp_gaz_sent", "t5_asp_lownergaz_sent"
        ])
    }
    sorted_df = df[["model", *columns]].sort_values(
        "model", key=lambda x: x.apply(lambda y: model_order.get(y, 1000)))

    # rename models to latex names
    model_names = {
        "flair_roberta": "FLAIR\textsubscript{RoBERTa-Large}",
        "t5_asp": "T5-ASP",
        "dict_match_gaz": "DictMatch\textsubscript{Gaz}",
        "dict_match_sent": "DictMatch\textsubscript{Sent}",
        "dict_match_lownergaz": "DictMatch\textsubscript{LownerGaz}",
        "dict_match_lownergaz_sent": "DictMatch\textsubscript{LownerGaz+Sent}",
        "dict_match_gaz_sent": "DictMatch\textsubscript{Gaz+Sent}",
        "t5_asp_gaz": "T5-ASP\textsubscript{Gaz}",
        "t5_asp_sent": "T5-ASP\textsubscript{Sent}",
        "t5_asp_lownergaz": "T5-ASP\textsubscript{LownerGaz}",
        "t5_asp_gaz_sent": "T5-ASP\textsubscript{Gaz+Sent}",
        "t5_asp_lownergaz_sent": "T5-ASP\textsubscript{LownerGaz+Sent}"
    }
    sorted_df["Models"] = sorted_df["model"].apply(lambda x: model_names[x])
    # combine mean + std in one column
    column_depth = sorted_df.columns.nlevels
    if column_depth == 1:
        for column, new_column in column_names.items():
            sorted_df[new_column] = sorted_df[column].map('{:,.4f}'.format)
        result_df = sorted_df[["Models", *list(column_names.values())
                                   ]].set_index("Models")
    elif column_depth == 2:
        for column, new_column in column_names.items():
            sorted_df[new_column] = sorted_df[column]['mean'].map(
                '{:,.2f}'.format) + " (" + (round(
                    sorted_df[column]['std'] *
                    100)).astype(int).astype(str) + ")"
        result_df = sorted_df[["Models", *list(column_names.values())
                               ]].set_index("Models")
        result_df.columns = ["".join(col) for col in result_df.columns.values]
    elif column_depth == 3:
        # iterate over column names - entities
        entity_order = {
            key: idx
            for idx, key in enumerate([
                "person", "location", "group", "corporation", "creative-work",
                "product"
            ])
        }
        entity_names = {
            "person": "Person",
            "location": "Location",
            "group": "Group",
            "corporation": "Corporation",
            "creative-work": "Creative Work",
            "product": "Product"
        }
        for level_0_col, ent_name in entity_names.items():
            for column, new_column in column_names.items():
                sorted_df[ent_name, new_column,
                          ""] = sorted_df[level_0_col][column]['mean'].map(
                              '{:,.2f}'.format) + " (" + (round(
                                  sorted_df[level_0_col][column]['std'] *
                                  100)).astype(int).astype(str) + ")"
        result_df = sorted_df[[
            ("Models", "", ""),
            *list(product(entity_names.values(), column_names.values(), [""]))
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