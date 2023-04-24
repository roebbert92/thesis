import sys
import os
from typing import Optional

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from finetuning.experiment_gazetteers import tune_gazetteers
from finetuning.experiment_sentences import tune_sentences


tune_sentences()
tune_gazetteers()
