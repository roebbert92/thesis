from itertools import product
from flair.models import SequenceTagger
from flair.data import Sentence
import flair
import torch
import os
from pathlib import Path
from typing import List
import random
import numpy as np
import copy
from nlpaug.augmenter.char import KeyboardAug
from tqdm import tqdm


def freeze(o):
    if isinstance(o, dict):
        return frozenset({k: freeze(v) for k, v in o.items()}.items())

    if isinstance(o, (set, tuple, list)):
        return tuple([freeze(v) for v in o])

    return o


def make_hash(o):
    """
    makes a hash out of anything that contains only list,dict and hashable types including string and numeric types
    """
    return hash(freeze(o))


def samples_are_equal(sample_a: dict, sample_b: dict):
    return make_hash(sample_a) == make_hash(sample_b)


keyboard_aug = KeyboardAug(
    aug_char_p=0.2,
    aug_char_max=3,
    aug_word_p=0.05,
    aug_word_max=6,
    tokenizer=lambda x: x.split(" "),
    reverse_tokenizer=lambda x: " ".join(x),
    include_numeric=False,
    include_special_char=False,
    include_upper_case=False,
)
# No flair for pytorch version 2.0+
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
flair.cache_root = Path(os.path.dirname(os.path.realpath(__file__)))
tagger = SequenceTagger.load("flair/pos-english")


def get_random_mask(size: int):
    if size == 0:
        return []
    mask = np.random.randint(2, size=size)
    if mask.sum() == 0:
        mask[np.random.randint(size)] = 1
    return mask.tolist()


def error_type1_augmentation(sample: dict, types: List[str]):
    augmented_sample = copy.deepcopy(sample)
    entity_spans = {(ent["start"], ent["end"]) for ent in augmented_sample["entities"]}
    # false positive: Take unlabeled noun phrases and randomly choose a label
    sent = Sentence(augmented_sample["tokens"])
    tagger.predict(sent)
    phrases = set()
    current_phrase = {"start": -1, "end": -1}
    for token in sent.get_labels("pos"):
        if "NN" in token.value:
            if current_phrase["start"] == -1:
                current_phrase["start"] = token.data_point.idx - 1
            current_phrase["end"] = token.data_point.idx
        elif current_phrase["end"] > -1:
            phrases.add((current_phrase["start"], current_phrase["end"]))
            current_phrase = {"start": -1, "end": -1}
    if current_phrase["end"] > -1:
        phrases.add((current_phrase["start"], current_phrase["end"]))

    valid_spans = list(phrases.difference(entity_spans))
    # make sure: no overlaps:
    for span in valid_spans:
        for ent_span in entity_spans:
            if span[0] <= ent_span[0] <= span[1] or span[1] >= ent_span[1] >= span[0]:
                valid_spans.remove(span)
                break

    if len(valid_spans) > 0:
        span = random.choice(valid_spans)
        augmented_sample["entities"].append(
            {
                "start": span[0],
                "end": span[1],
                "type": random.choice(types),
                "error": 1,
            }
        )
        augmented_sample["entities"] = sorted(
            augmented_sample["entities"], key=lambda ent: ent["start"]
        )
        return augmented_sample, True

    return augmented_sample, False


def has_valid_entity_idx(sample: dict, entity_idx: int):
    if len(sample["entities"]) == 0:
        return entity_idx, False
    while (
        len(sample["entities"]) <= entity_idx
        or "error" in sample["entities"][entity_idx]
    ):
        entity_idx -= 1
        if entity_idx == -1:
            return entity_idx, False
    return entity_idx, True


def error_type2_augmentation(sample: dict, types: List[str], entity_idx: int = 0):
    augmented_sample = copy.deepcopy(sample)
    # not labeled (false negative): remove label
    entity_idx, valid = has_valid_entity_idx(sample, entity_idx)
    if not valid:
        return augmented_sample, False
    augmented_sample["entities"][entity_idx]["error"] = 2
    return augmented_sample, True


def error_type3_augmentation(sample: dict, types: List[str], entity_idx: int = 0):
    # wrong labels, correct span: randomize label
    augmented_sample = copy.deepcopy(sample)
    type_set = set(types)
    entity_idx, valid = has_valid_entity_idx(sample, entity_idx)
    if not valid:
        return augmented_sample, False
    entity = augmented_sample["entities"][entity_idx]
    valid_types = list(type_set.difference([entity["type"]]))
    augmented_sample["entities"][entity_idx]["type"] = random.choice(valid_types)
    augmented_sample["entities"][entity_idx]["error"] = 3
    return augmented_sample, True


def error_type4_augmentation(
    sample: dict, types: List[str], avg_entity_length: int = 3, entity_idx: int = 0
):
    augmented_sample = copy.deepcopy(sample)
    max_end = len(augmented_sample["tokens"])
    min_start = 0
    type_set = set(types)
    action = {"increase", "decrease"}
    direction = {"left", "right", "both"}

    def apply_left_action(ents: List[dict], ent_idx: int, ent: dict, action: str):
        if action == "increase":
            # get left border
            left_space = 0
            if ent_idx > 0:
                left_space = ent["start"] - ents[ent_idx - 1]["end"] - 1
            else:
                left_space = ent["start"]
            left_space = min(left_space, avg_entity_length)
            if left_space > 0:
                ent["start"] -= random.randint(1, left_space)
                ent["error"] = 4
        elif action == "decrease":
            # get entity end as right border
            right_space = ent["end"] - ent["start"] - 1
            if right_space > 0:
                ent["start"] += random.randint(1, right_space)
                ent["error"] = 4

    def apply_right_action(ents: List[dict], ent_idx: int, ent: dict, action: str):
        if action == "increase":
            # get right border
            right_space = 0
            if ent_idx < len(ents) - 1:
                right_space = ents[ent_idx + 1]["start"] - ent["end"] - 1
            else:
                right_space = max_end - ent["end"]
            right_space = min(right_space, avg_entity_length)
            if right_space > 0:
                ent["end"] += random.randint(1, right_space) if right_space > 0 else 0
                ent["error"] = 4
        elif action == "decrease":
            # get entity start as right border
            left_space = ent["end"] - ent["start"] - 1
            if left_space > 0:
                ent["end"] -= random.randint(1, left_space)
                ent["error"] = 4

    entity_idx, valid = has_valid_entity_idx(sample, entity_idx)
    if not valid:
        return augmented_sample, False
    entity = augmented_sample["entities"][entity_idx]
    entity_length = entity["end"] - entity["start"]
    valid_actions = copy.deepcopy(action)
    valid_directions = copy.deepcopy(direction)
    if entity_length == 1:
        # increase only
        valid_actions.remove("decrease")
    valid_action_directions = set(product(list(valid_directions), list(valid_actions)))
    if (
        entity_idx > 0
        and augmented_sample["entities"][entity_idx - 1]["end"] == entity["start"] - 1
    ) or entity["start"] == min_start:
        # right only
        valid_action_directions.remove(("left", "increase"))
        valid_action_directions.remove(("both", "increase"))
    if (
        entity_idx < len(augmented_sample["entities"]) - 1
        and augmented_sample["entities"][entity_idx + 1]["start"] == entity["end"] + 1
    ) or entity["end"] == max_end:
        # left only
        valid_action_directions.remove(("right", "increase"))
        if ("both", "increase") in valid_action_directions:
            valid_action_directions.remove(("both", "increase"))

    if len(valid_action_directions) > 0:
        direction_chosen, action_chosen = random.choice(list(valid_action_directions))
        if direction_chosen == "both":
            apply_left_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
            apply_right_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
        elif direction_chosen == "left":
            apply_left_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
        elif direction_chosen == "right":
            apply_right_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )

    # randomize label
    if not samples_are_equal(sample, augmented_sample):
        valid_types = list(type_set.difference([entity["type"]]))
        entity["type"] = random.choice(valid_types)
        entity["error"] = 4
        return augmented_sample, True

    return augmented_sample, False


def error_type5_augmentation(
    sample: dict, types: List[str], avg_entity_length: int = 3, entity_idx: int = 0
):
    augmented_sample = copy.deepcopy(sample)
    # Overlapping span, correct label: if len == 1, increase left or right, else increase/decrease left or right (depending on sentence length)
    max_end = len(augmented_sample["tokens"])
    min_start = 0
    action = {"increase", "decrease"}
    direction = {"left", "right", "both"}

    def apply_left_action(ents: List[dict], ent_idx: int, ent: dict, action: str):
        if action == "increase":
            # get left border
            left_space = 0
            if ent_idx > 0:
                left_space = ent["start"] - ents[ent_idx - 1]["end"] - 1
            else:
                left_space = ent["start"]
            left_space = min(left_space, avg_entity_length)
            if left_space > 0:
                ent["start"] -= random.randint(1, left_space)
                ent["error"] = 5
        elif action == "decrease":
            # get entity end as right border
            right_space = ent["end"] - ent["start"] - 1
            if right_space > 0:
                ent["start"] += random.randint(1, right_space)
                ent["error"] = 5

    def apply_right_action(ents: List[dict], ent_idx: int, ent: dict, action: str):
        if action == "increase":
            # get right border
            right_space = 0
            if ent_idx < len(ents) - 1:
                right_space = ents[ent_idx + 1]["start"] - ent["end"] - 1
            else:
                right_space = max_end - ent["end"]
            right_space = min(right_space, avg_entity_length)
            if right_space > 0:
                ent["end"] += random.randint(1, right_space) if right_space > 0 else 0
                ent["error"] = 5
        elif action == "decrease":
            # get entity start as right border
            left_space = ent["end"] - ent["start"] - 1
            if left_space > 0:
                ent["end"] -= random.randint(1, left_space)
                ent["error"] = 5

    entity_idx, valid = has_valid_entity_idx(sample, entity_idx)
    if not valid:
        return augmented_sample, False
    entity = augmented_sample["entities"][entity_idx]
    entity_length = entity["end"] - entity["start"]
    valid_actions = copy.deepcopy(action)
    valid_directions = copy.deepcopy(direction)
    if entity_length == 1:
        # increase only
        valid_actions.remove("decrease")
    valid_action_directions = set(product(list(valid_directions), list(valid_actions)))
    if (
        entity_idx > 0
        and augmented_sample["entities"][entity_idx - 1]["end"] == entity["start"] - 1
    ) or entity["start"] == min_start:
        # right only
        valid_action_directions.remove(("left", "increase"))
        valid_action_directions.remove(("both", "increase"))
    if (
        entity_idx < len(augmented_sample["entities"]) - 1
        and augmented_sample["entities"][entity_idx + 1]["start"] == entity["end"] + 1
    ) or entity["end"] == max_end:
        # left only
        valid_action_directions.remove(("right", "increase"))
        if ("both", "increase") in valid_action_directions:
            valid_action_directions.remove(("both", "increase"))

    if len(valid_action_directions) > 0:
        direction_chosen, action_chosen = random.choice(list(valid_action_directions))
        if direction_chosen == "both":
            apply_left_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
            apply_right_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
        elif direction_chosen == "left":
            apply_left_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
        elif direction_chosen == "right":
            apply_right_action(
                augmented_sample["entities"],
                entity_idx,
                entity,
                action_chosen,
            )
    return augmented_sample, not samples_are_equal(sample, augmented_sample)


def spelling_error_augmentation(sample: dict, types: List[str]):
    augmented_sample = copy.deepcopy(sample)
    augmented_sample["tokens"] = keyboard_aug.augment(
        " ".join(augmented_sample["tokens"])
    )[0].split(" ")
    if not samples_are_equal(sample, augmented_sample):
        augmented_sample["extended"] = augmented_sample["tokens"]
        augmented_sample["error"] = 0
        return augmented_sample, True
    return augmented_sample, False


def spelling_no_augmentation(sample: dict, types: List[str]):
    return sample, True


def make_erroneous_dataset(
    dataset: List[dict],
    types: List[str],
    ratio: float,
    error_dist: List[float] = [0.36, 0.5, 0.04, 0.01, 0.09],
    spelling_dist: List[float] = [0.3],
):
    result_dataset = copy.deepcopy(dataset)
    samples = [sample_idx for sample_idx, _ in enumerate(dataset)]
    error_size_samples = round(
        len([1 for sample in dataset for _ in sample["entities"]]) * ratio
    )
    type_list = list(types)

    error_type1_size = round(error_size_samples * error_dist[0])
    error_type1_samples = 0
    with tqdm(total=error_type1_size, desc="adding error_type1") as pbar:
        while error_type1_samples < error_type1_size:
            sample_idx = random.choice(samples)
            aug_sample, augmented = error_type1_augmentation(
                result_dataset[sample_idx], type_list
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type1_samples += 1
                pbar.update()

    error_type4_size = round(error_size_samples * error_dist[3])
    error_type4_samples = 0
    with tqdm(total=error_type4_size, desc="adding error_type4") as pbar:
        while error_type4_samples < error_type4_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            possible_entities = [
                ent_idx
                for ent_idx, ent in enumerate(result_dataset[sample_idx]["entities"])
                if "error" not in ent
            ]
            if len(possible_entities) > 0:
                entity_idx = random.choice(possible_entities)
                aug_sample, augmented = error_type4_augmentation(
                    result_dataset[sample_idx], type_list, entity_idx=entity_idx
                )
                if augmented:
                    result_dataset[sample_idx] = aug_sample
                    error_type4_samples += 1
                    pbar.update()

    error_type5_size = round(error_size_samples * error_dist[4])
    error_type5_samples = 0
    with tqdm(total=error_type5_size, desc="adding error_type5") as pbar:
        while error_type5_samples < error_type5_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type5_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type5_samples += 1
                pbar.update()

    error_type3_size = round(error_size_samples * error_dist[2])
    error_type3_samples = 0
    with tqdm(total=error_type3_size, desc="adding error_type3") as pbar:
        while error_type3_samples < error_type3_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type3_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type3_samples += 1
                pbar.update()

    error_type2_size = round(error_size_samples * error_dist[1])
    error_type2_samples = 0
    with tqdm(total=error_type2_size, desc="adding error_type2") as pbar:
        while error_type2_samples < error_type2_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type2_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type2_samples += 1
                pbar.update()

    print("adding spelling error")
    spelling_error_samples = np.random.choice(
        [idx for idx in samples],
        size=round(error_size_samples * spelling_dist[0]),
    ).tolist()
    for idx in spelling_error_samples:
        result_dataset[idx], _ = spelling_error_augmentation(
            result_dataset[idx], type_list
        )
    return result_dataset


def make_erroneous_gazetteer(
    dataset: List[dict],
    types: List[str],
    ratio: float,
    error_dist: List[float] = [0.5, 0.16, 0.13, 0.21],
    spelling_dist: List[float] = [0.3],
):
    result_dataset = copy.deepcopy(dataset)
    samples = [sample_idx for sample_idx, _ in enumerate(dataset)]
    error_size_samples = round(
        len([1 for sample in dataset for _ in sample["entities"]]) * ratio
    )
    type_list = list(types)

    error_type4_size = round(error_size_samples * error_dist[2])
    error_type4_samples = 0
    with tqdm(total=error_type4_size, desc="adding error_type4") as pbar:
        while error_type4_samples < error_type4_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type4_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type4_samples += 1
                pbar.update()

    error_type5_size = round(error_size_samples * error_dist[3])
    error_type5_samples = 0
    with tqdm(total=error_type5_size, desc="adding error_type5") as pbar:
        while error_type5_samples < error_type5_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type5_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type5_samples += 1
                pbar.update()

    error_type3_size = round(error_size_samples * error_dist[1])
    error_type3_samples = 0
    with tqdm(total=error_type3_size, desc="adding error_type3") as pbar:
        while error_type3_samples < error_type3_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type3_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type3_samples += 1
                pbar.update()

    error_type2_size = round(error_size_samples * error_dist[0])
    error_type2_samples = 0
    with tqdm(total=error_type2_size, desc="adding error_type2") as pbar:
        while error_type2_samples < error_type2_size:
            sample_idx = random.choice(samples)
            while len(result_dataset[sample_idx]["entities"]) == 0:
                sample_idx = random.choice(samples)
            entity_idx = random.choice(
                range(len(result_dataset[sample_idx]["entities"]))
            )
            aug_sample, augmented = error_type2_augmentation(
                result_dataset[sample_idx], type_list, entity_idx=entity_idx
            )
            if augmented:
                result_dataset[sample_idx] = aug_sample
                error_type2_samples += 1
                pbar.update()

    print("adding spelling error")
    spelling_error_samples = np.random.choice(
        [idx for idx in samples],
        size=round(error_size_samples * spelling_dist[0]),
    ).tolist()
    for idx in spelling_error_samples:
        result_dataset[idx], _ = spelling_error_augmentation(
            result_dataset[idx], type_list
        )
    return result_dataset
