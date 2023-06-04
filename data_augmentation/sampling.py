from typing import List
from collections import defaultdict, Counter
import random
import pandas as pd


def per_type_uniform_sampling(dataset: List[dict], types: List[str],
                              sample_size: int):
    # count entities per doc
    doc_entity_count = []
    for doc in dataset:
        doc_entity_count.append({
            "doc_id":
            doc["doc_id"],
            **Counter([ent["type"] for ent in doc["entities"]])
        })
    count_df = pd.DataFrame.from_records(doc_entity_count).fillna(0)

    # get types
    type_to_ids = {
        t: frozenset(count_df[count_df[t] > 0]["doc_id"].tolist())
        for t in types
    }
    type_count = defaultdict(int)
    sampled_ids = set()

    type_set = list(types)
    random.shuffle(type_set)

    for typ in type_set:
        done_sampling = False
        while not done_sampling:
            if type_count[typ] >= sample_size:
                done_sampling = True
                break
            sample_set = list(type_to_ids[typ].difference(sampled_ids))
            select_id = random.choice(sample_set)
            sampled_ids.add(select_id)
            entity_count = count_df[count_df["doc_id"] == select_id].iloc[0]
            for t in types:
                type_count[t] += entity_count[t]

    return [doc for doc in dataset if doc["doc_id"] in sampled_ids], type_count
