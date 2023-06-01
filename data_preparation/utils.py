from typing import List
import fasttext
import string
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat

fmodel = fasttext.load_model(
    "/home/loebbert/projects/thesis/data_preparation/lang_detect/lid.176.bin")
punct_removal = str.maketrans("", "", string.punctuation)

supported_lang = ["en", "fr", "de", "it", "es", "pt", "nl", "pl"]


def is_supported_doc(tokens: List[str]):
    text = " ".join(tokens)
    no_punct_text = text.translate(punct_removal)
    if len(no_punct_text.split()) > 0:
        # check supported language
        pred, _ = fmodel.predict(text)
        pred = str(pred[0]).removeprefix("__label__") # type: ignore
        if pred in supported_lang:
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
