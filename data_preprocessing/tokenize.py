import copy
import os
import json
from transformers import PreTrainedTokenizer

MENTION_START = "<m>"
MENTION_END = "</m>"


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    elif word == "''" or word == "``":  # <unk> otherwise
        return "\""
    elif word == "`":  # <unk> otherwise
        return "\'"
    else:
        return word


def is_punctuation(c):
    if (c in {
            ".", ",", "?", "!", ";", ":", "'s", "'m", "'ve", "n't", "'ll", ")",
            "}", "]"
    }):
        return True
    return False


def get_subtokens(tokenizer: PreTrainedTokenizer, word):
    word = normalize_word(word, "english")
    if word == "(" or word == "[":
        subtokens = tokenizer.tokenize(word)
    elif word in [")", "]", "\'"]:
        subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
    elif is_punctuation(word):
        subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
    else:
        subtokens = tokenizer.tokenize(word)
    return subtokens


def map_entities_to_sentence(entities, sentence, inv_subtoken_map,
                             subtoken_map, label_to_id):
    if len(entities) > 0:
        m_types = [x['type'] for x in entities]
        m_startings = [x['start'] for x in entities]
        m_endings = [x['end'] for x in entities]
    else:
        m_types, m_startings, m_endings = [], [], []

    sorted_pos = sorted(
        [(inv_subtoken_map[x][0], MENTION_END, label_to_id[t], idx)
         for idx, (x, t) in enumerate(zip(m_endings, m_types))] +
        [(inv_subtoken_map[x][0], MENTION_START, label_to_id[t], idx)
         for idx, (x, t) in enumerate(zip(m_startings, m_types))],
        reverse=True)

    target_sentence = copy.deepcopy(sentence)
    ent_indices = [-1 for _ in range(len(sentence))]
    ent_type_sequence = [-1 for _ in range(len(sentence))]
    target_subtoken_map = copy.deepcopy(subtoken_map)

    # end_to_index_in_target = {}

    for (subtoken_idx, mention_type, label_id, entity_idx) in sorted_pos:

        target_sentence.insert(subtoken_idx,
                               mention_type)  # insert end or start
        # insert pairing bracket index for entity
        ent_indices.insert(subtoken_idx, entity_idx)

        if mention_type == MENTION_END:  # insert entity type
            ent_type_sequence.insert(subtoken_idx, label_id)
        else:
            ent_type_sequence.insert(subtoken_idx, -1)

        # for k in end_to_index_in_target:  # map index in src to index in target
        #     # plus 1 for every special token inserted
        #     end_to_index_in_target[k] += 1
        # end_to_index_in_target[subtoken_idx] = subtoken_idx

        if mention_type == MENTION_END:
            target_subtoken_map.insert(subtoken_idx,
                                       subtoken_map[subtoken_idx - 1])
        elif mention_type == MENTION_START:
            target_subtoken_map.insert(subtoken_idx,
                                       subtoken_map[subtoken_idx + 1])

    return (
        target_sentence,
        ent_indices,
        ent_type_sequence,
        # end_to_index_in_target,
        target_subtoken_map)


def get_target_sentence(tokenizer: PreTrainedTokenizer, label_to_id, tokens,
                        entities):
    processed, subtoken_map, inv_subtoken_map = [], [], {}

    for word_idx, word in enumerate(tokens):
        # no prefix inserted here
        subtokens = get_subtokens(tokenizer, word)
        inv_subtoken_map[word_idx] = (len(processed),
                                      len(processed) + len(subtokens))

        for subtoken in subtokens:
            processed.append(subtoken)
            subtoken_map.append(word_idx)

    inv_subtoken_map[len(tokens)] = (len(processed), len(processed) + 1)
    processed.append(tokenizer.eos_token)
    subtoken_map.append(len(tokens))

    target_sentence, ent_indices, ent_type_sequence, _ = map_entities_to_sentence(
        entities, processed, inv_subtoken_map, subtoken_map, label_to_id)
    return (processed, target_sentence, ent_type_sequence, ent_indices,
            subtoken_map)


def get_input_sentence(tokenizer: PreTrainedTokenizer,
                       doc,
                       insert_prefix=True):
    processed_doc, subtoken_map = [], []
    first_token_in_doc = True
    for word_idx, word in enumerate(doc):
        if first_token_in_doc and insert_prefix:
            # insert prefix
            prefix_text = tokenizer.tokenize("named entity recognition:")
            for subtoken in prefix_text:
                processed_doc.append(subtoken)
                subtoken_map.append(word_idx)

        subtokens = get_subtokens(tokenizer, word)
        for subtoken in subtokens:
            processed_doc.append(subtoken)
            subtoken_map.append(word_idx)

        first_token_in_doc = False
    processed_doc.append(tokenizer.eos_token)
    subtoken_map.append(len(doc))

    return processed_doc


def tokenize_json(tokenizer: PreTrainedTokenizer,
                  train_file,
                  dev_file,
                  test_file,
                  type_file,
                  prepend_task_description=True):
    tokenizer.add_tokens(MENTION_START)
    tokenizer.add_tokens(MENTION_END)

    with open(type_file, encoding="utf-8") as file:
        labels = json.load(file)['entities']
    label_to_id = {label: id for id, label in enumerate(labels)}

    for file_name in [train_file, dev_file, test_file]:
        tokenized_dataset = []
        with open(file_name, encoding="utf-8") as file:
            instances = json.load(file)

        name = os.path.basename(os.path.splitext(file_name)[0])
        dropped = []

        for inst_id, instance in enumerate(instances):
            tokens = instance['tokens']
            entities = instance['entities']
            extended = instance['extended']

            tokenized_sentence, target_sentence, entity_type_sequence, entity_indices, subtoken_map = get_target_sentence(
                tokenizer, label_to_id, tokens, entities)

            # insert prefix (instruction for model) here
            input_sentence = get_input_sentence(tokenizer, extended,
                                                prepend_task_description)
            if len(tokenized_sentence) > 256 or len(input_sentence) > 256:
                dropped.append(str(inst_id))
                continue
            tokenized_dataset.append({
                "doc_id": name + "_" + str(inst_id),
                "sentence": tokenized_sentence,
                # sentence is for copy mechanism, might be different from
                # input_sentence which is for encoding only
                "input_sentence": input_sentence,
                "target_sentence": target_sentence,
                "subtoken_map": subtoken_map,
                "ent_type_sequence": entity_type_sequence,
                "ent_indices": entity_indices
            })

        print(f"{name}: dropped {len(dropped)} inst_ids: " + " ".join(dropped))
        with open(
                f"{os.path.splitext(file_name)[0]}.{os.path.basename(os.path.splitext(tokenizer.name_or_path)[0])}.jsonlines",
                "w",
                encoding="utf-8") as output_file:
            json.dump(tokenized_dataset, output_file)
