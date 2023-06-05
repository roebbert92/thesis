from flair.models import SequenceTagger
from flair.data import Sentence
from typing import List
import random
import numpy as np
import copy
from nlpaug.augmenter.char import KeyboardAug

keyboard_aug = KeyboardAug(aug_char_p=0.2,
                           aug_char_max=3,
                           aug_word_p=0.05,
                           aug_word_max=6,
                           tokenizer=lambda x: x.split(" "),
                           reverse_tokenizer=lambda x: " ".join(x),
                           include_numeric=False,
                           include_special_char=False,
                           include_upper_case=False)

tagger = SequenceTagger.load("flair/pos-english-fast")


def get_random_mask(size: int):
    if size == 0:
        return []
    mask = np.random.randint(2, size=size)
    if mask.sum() == 0:
        mask[np.random.randint(size)] = 1
    return mask.tolist()


def error_type1_augmentation(sample: dict, types: List[str]):
    augmented_sample = copy.deepcopy(sample)
    entity_spans = {(ent["start"], ent["end"])
                    for ent in augmented_sample["entities"]}
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
    mask = get_random_mask(len(valid_spans))
    for span, valid in zip(valid_spans, mask):  # type: ignore
        if valid:
            augmented_sample["entities"].append({
                "start": span[0],
                "end": span[1],
                "type": random.choice(types)
            })

    return augmented_sample


def error_type2_augmentation(sample: dict, types: List[str]):
    augmented_sample = copy.deepcopy(sample)
    # not labeled (false negative): remove label
    mask = get_random_mask(len(augmented_sample["entities"]))
    augmented_sample["entities"] = [
        ent for valid, ent in zip(mask, augmented_sample["entities"])
        if not valid
    ]
    return augmented_sample


def error_type3_augmentation(sample: dict, types: List[str]):
    # wrong labels, correct span: randomize label
    augmented_sample = copy.deepcopy(sample)
    mask = get_random_mask(len(augmented_sample["entities"]))
    type_set = set(types)
    for valid, entity in zip(mask, augmented_sample["entities"]):
        if valid:
            valid_types = list(type_set.difference([entity["type"]]))
            entity["type"] = random.choice(valid_types)
    return augmented_sample


def error_type4_augmentation(sample: dict,
                             types: List[str],
                             avg_entity_length: int = 3):
    augmented_sample = copy.deepcopy(sample)
    max_end = len(augmented_sample["tokens"])
    min_start = 0
    mask = get_random_mask(len(augmented_sample["entities"]))
    augmented_sample["entities"] = sorted(augmented_sample["entities"],
                                          key=lambda x: x["start"])
    type_set = set(types)
    action = {"increase", "decrease"}
    direction = {"left", "right", "both"}

    def apply_left_action(ents: List[dict], ent_idx: int, ent: dict,
                          action: str):
        if action == "increase":
            # get left border
            left_space = 0
            if ent_idx > 0:
                left_space = ent["start"] - ents[ent_idx - 1]["end"]
            else:
                left_space = ent["start"]
            left_space = min(left_space, avg_entity_length)
            ent["start"] -= random.randint(1,
                                           left_space) if left_space > 0 else 0
        elif action == "decrease":
            # get entity end as right border
            right_space = ent["end"] - ent["start"] - 1
            ent["start"] += random.randint(
                1, right_space) if right_space > 0 else 0

    def apply_right_action(ents: List[dict], ent_idx: int, ent: dict,
                           action: str):
        if action == "increase":
            # get right border
            right_space = 0
            if ent_idx < len(ents) - 1:
                right_space = ents[ent_idx + 1]["start"] - ent["end"]
            else:
                right_space = max_end - ent["end"]
            right_space = min(right_space, avg_entity_length)
            ent["end"] += random.randint(1,
                                         right_space) if right_space > 0 else 0
        elif action == "decrease":
            # get entity start as right border
            left_space = ent["end"] - ent["start"] - 1
            ent["end"] -= random.randint(1,
                                         left_space) if left_space > 0 else 0

    for valid, entity_idx, entity in zip(
            mask,
            range(len(augmented_sample["entities"])),  # type: ignore
            augmented_sample["entities"]):
        if valid:
            entity_length = entity["end"] - entity["start"]
            valid_actions = copy.deepcopy(action)
            valid_directions = copy.deepcopy(direction)
            if entity_length == 1:
                # increase only
                valid_actions.remove("decrease")
            if (entity_idx > 0
                    and augmented_sample["entities"][entity_idx - 1]["end"]
                    == entity["start"] - 1) or entity["start"] == min_start:
                # right only
                valid_directions.remove("left")
                valid_directions.remove("both")
            if (entity_idx < len(augmented_sample["entities"]) - 1
                    and augmented_sample["entities"][entity_idx + 1]["start"]
                    == entity["end"] + 1) or entity["end"] == max_end:
                # left only
                valid_directions.remove("right")
                if "both" in valid_directions:
                    valid_directions.remove("both")

            if len(valid_actions) > 0 and len(valid_directions) > 0:
                direction_chosen = random.choice(list(valid_directions))
                if direction_chosen == "both":
                    apply_left_action(augmented_sample["entities"],
                                      entity_idx, entity,
                                      random.choice(list(valid_actions)))
                    apply_right_action(augmented_sample["entities"],
                                       entity_idx, entity,
                                       random.choice(list(valid_actions)))
                elif direction_chosen == "left":
                    apply_left_action(augmented_sample["entities"],
                                      entity_idx, entity,
                                      random.choice(list(valid_actions)))
                elif direction_chosen == "right":
                    apply_right_action(augmented_sample["entities"],
                                       entity_idx, entity,
                                       random.choice(list(valid_actions)))

            # randomize label
            valid_types = list(type_set.difference([entity["type"]]))
            entity["type"] = random.choice(valid_types)

    return augmented_sample


def error_type5_augmentation(sample: dict,
                             types: List[str],
                             avg_entity_length: int = 3):
    augmented_sample = copy.deepcopy(sample)
    # Overlapping span, correct label: if len == 1, increase left or right, else increase/decrease left or right (depending on sentence length)
    max_end = len(augmented_sample["tokens"])
    min_start = 0
    mask = get_random_mask(len(augmented_sample["entities"]))
    augmented_sample["entities"] = sorted(augmented_sample["entities"],
                                          key=lambda x: x["start"])
    action = {"increase", "decrease"}
    direction = {"left", "right", "both"}

    def apply_left_action(ents: List[dict], ent_idx: int, ent: dict,
                          action: str):
        if action == "increase":
            # get left border
            left_space = 0
            if ent_idx > 0:
                left_space = ent["start"] - ents[ent_idx - 1]["end"]
            else:
                left_space = ent["start"]
            left_space = min(left_space, avg_entity_length)
            ent["start"] -= random.randint(1,
                                           left_space) if left_space > 0 else 0
        elif action == "decrease":
            # get entity end as right border
            right_space = ent["end"] - ent["start"] - 1
            ent["start"] += random.randint(
                1, right_space) if right_space > 0 else 0

    def apply_right_action(ents: List[dict], ent_idx: int, ent: dict,
                           action: str):
        if action == "increase":
            # get right border
            right_space = 0
            if ent_idx < len(ents) - 1:
                right_space = ents[ent_idx + 1]["start"] - ent["end"]
            else:
                right_space = max_end - ent["end"]
            right_space = min(right_space, avg_entity_length)
            ent["end"] += random.randint(1,
                                         right_space) if right_space > 0 else 0
        elif action == "decrease":
            # get entity start as right border
            left_space = ent["end"] - ent["start"] - 1
            ent["end"] -= random.randint(1,
                                         left_space) if left_space > 0 else 0

    for valid, entity_idx, entity in zip(
            mask,
            range(len(augmented_sample["entities"])),  # type: ignore
            augmented_sample["entities"]):
        if valid:
            entity_length = entity["end"] - entity["start"]
            valid_actions = copy.deepcopy(action)
            valid_directions = copy.deepcopy(direction)
            if entity_length == 1:
                # increase only
                valid_actions.remove("decrease")
            if (entity_idx > 0
                    and augmented_sample["entities"][entity_idx - 1]["end"]
                    == entity["start"] - 1) or entity["start"] == min_start:
                # right only
                valid_directions.remove("left")
                valid_directions.remove("both")
            if (entity_idx < len(augmented_sample["entities"]) - 1
                    and augmented_sample["entities"][entity_idx + 1]["start"]
                    == entity["end"] + 1) or entity["end"] == max_end:
                # left only
                valid_directions.remove("right")
                if "both" in valid_directions:
                    valid_directions.remove("both")

            if len(valid_actions) > 0 and len(valid_directions) > 0:
                direction_chosen = random.choice(list(valid_directions))
                if direction_chosen == "both":
                    apply_left_action(augmented_sample["entities"],
                                      entity_idx, entity,
                                      random.choice(list(valid_actions)))
                    apply_right_action(augmented_sample["entities"],
                                       entity_idx, entity,
                                       random.choice(list(valid_actions)))
                elif direction_chosen == "left":
                    apply_left_action(augmented_sample["entities"],
                                      entity_idx, entity,
                                      random.choice(list(valid_actions)))
                elif direction_chosen == "right":
                    apply_right_action(augmented_sample["entities"],
                                       entity_idx, entity,
                                       random.choice(list(valid_actions)))
    return augmented_sample


def spelling_error_augmentation(sample: dict, types: List[str]):
    augmented_sample = copy.deepcopy(sample)
    augmented_sample["tokens"] = keyboard_aug.augment(" ".join(
        augmented_sample["tokens"]))[0].split(" ")
    return augmented_sample


def make_erroneous_dataset(dataset: List[dict], types: List[str],
                           ratio: float):
    result_dataset = copy.deepcopy(dataset)
    mask = np.zeros(len(dataset))
    error_size = round(len(dataset) * ratio) + 1
    mask[:error_size] = 1
    np.random.shuffle(mask)

    type_list = list(types)

    # augmentations
    augmentations = [
        error_type1_augmentation, error_type2_augmentation,
        error_type3_augmentation, error_type4_augmentation,
        error_type5_augmentation, spelling_error_augmentation
    ]

    for valid, sample_idx, sample in zip(mask, range(len(dataset)), dataset):
        if valid:
            mask_augmentations = get_random_mask(len(augmentations))
            augmented_sample = copy.deepcopy(sample)
            for valid_aug, aug in zip(mask_augmentations, augmentations):
                if valid_aug:
                    augmented_sample = aug(augmented_sample, type_list)
            result_dataset[sample_idx] = augmented_sample

    return result_dataset