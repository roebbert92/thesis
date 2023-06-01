from typing import List
import fasttext
import string

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
        pred = str(pred[0]).removeprefix("__label__")
        if pred in supported_lang:
            return True
    return False
