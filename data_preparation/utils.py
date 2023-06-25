import os
from typing import List
import fasttext
import string
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat
import json
from seqscore.conll import ingest_conll_file, write_docs_using_encoding
from pathlib import Path

# fmodel = fasttext.load_model(
#     "/home/loebbert/projects/thesis/data_preparation/lang_detect/lid.176.bin")
punct_removal = str.maketrans("", "", string.punctuation)

supported_lang = ["en", "fr", "de", "it", "es", "pt", "nl", "pl"]


def is_supported_doc(tokens: List[str]):
    text = " ".join(tokens)
    no_punct_text = text.translate(punct_removal)
    if len(no_punct_text.split()) > 0:
        # check supported language
        # pred, _ = fmodel.predict(text, k=3)
        # pred = [str(p).removeprefix("__label__") for p in pred]  # type: ignore
        # for p in pred:
        #     if p in supported_lang:
        #         return True
        return True
    return False


def filter(d: List[dict], other_texts: List[str]):
    return [doc for doc in d if " ".join(doc["tokens"]) not in other_texts]


def remove_exact_matches(dataset: List[dict], other: List[dict]):
    other_text = [" ".join(doc["tokens"]) for doc in other]

    filtered_dataset = []
    chunk_size = len(dataset) // mp.cpu_count() + 1
    chunks = [
        dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        for res in pool.starmap(filter, zip(chunks, repeat(other_text))):
            filtered_dataset.extend(res)

    return filtered_dataset


def to_bio(items: List[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as file:
        for item in items:
            entities = {ent["start"]: ent for ent in item["entities"]}
            current_entity = None
            for idx, token in enumerate(item["tokens"]):
                if current_entity is None:
                    # check if new entity starts
                    if idx in entities:
                        current_entity = entities[idx]

                # process token
                label = "O"
                if current_entity is not None:
                    pos = "I"
                    if idx == current_entity["start"]:
                        pos = "B"
                    label = "-".join([pos, current_entity["type"]])
                line = " ".join([token, label]) + "\n"
                file.write(line)

                # check if entity ended
                if current_entity is not None:
                    if idx == current_entity["end"] - 1:
                        current_entity = None
            # write seperator line
            file.write("\n")
    return output_path


def json_to_bio(file_name: str):
    with open(file_name, "r", encoding="utf-8") as file:
        items = json.load(file)

    output_path = os.path.splitext(file_name)[0] + ".bio"
    return to_bio(items, output_path)


def convert_bio_to_bmes(file_name: str):
    input_labels = "BIO"
    output_labels = "BMES"
    file_encoding = "UTF-8"
    ignore_document_boundaries = True
    ignore_comment_lines = False
    docs = ingest_conll_file(
        file_name,
        input_labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )
    output_file_name = os.path.splitext(file_name)[0] + ".bmes"

    write_docs_using_encoding(docs, output_labels, file_encoding, " ",
                              output_file_name)
    return output_file_name
