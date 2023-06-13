import copy
import os
import json
from transformers import PreTrainedTokenizer
from haystack import Pipeline, Document
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

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

    try:
        sorted_pos = sorted(
            [(inv_subtoken_map[x][0], MENTION_END, label_to_id[t], idx)
             for idx, (x, t) in enumerate(zip(m_endings, m_types))] +
            [(inv_subtoken_map[x][0], MENTION_START, label_to_id[t], idx)
             for idx, (x, t) in enumerate(zip(m_startings, m_types))],
            reverse=True)
    except Exception as e:
        print(entities, sentence, inv_subtoken_map, subtoken_map, label_to_id)

        raise Exception(e)

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
                  file_name,
                  type_file,
                  output_path,
                  prepend_task_description=True):
    if MENTION_START not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_START)
    if MENTION_END not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_END)

    with open(type_file, encoding="utf-8") as file:
        labels = json.load(file)['entities']
    label_to_id = {label: id for id, label in enumerate(labels)}

    tokenized_dataset = []
    with open(file_name, encoding="utf-8") as file:
        instances = json.load(file)

    name = os.path.basename(os.path.splitext(file_name)[0])

    for inst_id, instance in tqdm(enumerate(instances),
                                  desc="Tokenization",
                                  total=len(instances)):
        tokens = instance['tokens']
        entities = instance['entities']
        extended = instance['extended']
        doc_id = instance[
            'doc_id'] if "doc_id" in instance else name + "_" + str(inst_id)

        tokenized_sentence, target_sentence, entity_type_sequence, entity_indices, subtoken_map = get_target_sentence(
            tokenizer, label_to_id, tokens, entities)

        # insert prefix (instruction for model) here
        input_sentence = get_input_sentence(tokenizer, extended,
                                            prepend_task_description)
        tokenized_dataset.append({
            "doc_id": doc_id,
            "sentence": tokenized_sentence,
            # sentence is for copy mechanism, might be different from
            # input_sentence which is for encoding only
            "input_sentence": input_sentence,
            "target_sentence": target_sentence,
            "subtoken_map": subtoken_map,
            "ent_type_sequence": entity_type_sequence,
            "ent_indices": entity_indices
        })

    output_file_name = os.path.join(
        output_path,
        f"{os.path.basename(os.path.splitext(file_name)[0])}.{os.path.basename(os.path.splitext(tokenizer.name_or_path)[0])}.jsonlines"
    )
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        json.dump(tokenized_dataset, output_file)
    return output_file_name


def handle_results(tokenizer: PreTrainedTokenizer, processed_doc: list,
                   results: List[Document], sent_use_labels: bool,
                   sent_use_mentions: bool, gaz_use_labels: bool,
                   gaz_use_mentions: bool):
    for result in results:
        meta = result.meta
        content = str(result.content)
        if meta["data_type"] == "gazetteers":
            text = tokenizer.tokenize(content)
            # if gaz_use_labels:
            #     tokenized_type = tokenizer.tokenize(":" + meta["type"])
            #     text.extend(tokenized_type)
            # if gaz_use_mentions:
            #     text = [MENTION_START, *text, MENTION_END]
            if gaz_use_mentions:
                text = [MENTION_START, *text, MENTION_END]
            if gaz_use_labels:
                tokenized_type = tokenizer.tokenize(":" + meta["type"])
                text.extend(
                    tokenized_type[1:] if gaz_use_mentions else tokenized_type)
            processed_doc.extend(text)
            processed_doc.append(tokenizer.eos_token)
        elif meta["data_type"] == "sentences":
            entity_starts = [entity["start"] for entity in meta["entities"]]
            entity_ends = {
                entity["end"]: entity
                for entity in meta["entities"]
            }
            for word_idx, word in enumerate(content.split()):
                # entity end
                if word_idx in entity_ends:
                    # if sent_use_labels:
                    #     tokenized_type = tokenizer.tokenize(
                    #         ":" + entity_ends[word_idx]["type"])
                    #     processed_doc.extend(tokenized_type)
                    # if sent_use_mentions:
                    #     processed_doc.append(MENTION_END)
                    if sent_use_mentions:
                        processed_doc.append(MENTION_END)
                    if sent_use_labels:
                        tokenized_type = tokenizer.tokenize(
                            ":" + entity_ends[word_idx]["type"])
                        processed_doc.extend(
                            tokenized_type[1:]
                            if sent_use_mentions else tokenized_type)
                # entity start
                if word_idx in entity_starts:
                    if sent_use_mentions:
                        processed_doc.append(MENTION_START)

                subtokens = get_subtokens(tokenizer, word)
                for subtoken in subtokens:
                    processed_doc.append(subtoken)
            processed_doc.append(tokenizer.eos_token)


def cosine_sim(a, b):
    same_size_a = np.tile(a, (b.shape[0], 1))
    np_array = np.sum(same_size_a * b,
                      axis=1) / (norm(same_size_a, axis=1) * norm(b, axis=1))
    return np_array.astype(float).tolist()


def get_embedding(cosine_model, embed_cache, input: str):
    if input not in embed_cache:
        embed_cache[input] = cosine_model.encode(input)
    return embed_cache[input]


def get_input_sentence_database_with_filter(
        tokenizer: PreTrainedTokenizer,
        doc_id,
        doc,
        database: Pipeline,
        cosine_model: Optional[SentenceTransformer],
        embed_cache,
        use_labels: bool,
        use_mentions: bool,
        filters: dict = {},
        filter_exact_match: bool = True,
        filter_same_document: bool = True,
        filtered_document_ids=[],
        prepend_examples=False,
        insert_prefix=True):
    sentence = " ".join(doc)
    exclude_filter = {}
    if filter_exact_match:
        exclude_filter["content"] = sentence
    if filter_same_document:
        exclude_filter["doc_id"] = [doc_id]
    if len(filtered_document_ids) > 0:
        exclude_filter["_id"] = filtered_document_ids

    results = database.run(
        query=sentence,
        params={"filters": {
            "$and": {
                "$not": exclude_filter,
                **filters
            }
        }})
    results = results["documents"] if results is not None else []
    similarities = []
    if len(results) > 0 and cosine_model is not None:
        if results[0].embedding is not None:
            sent_embed = get_embedding(cosine_model, embed_cache, sentence)
            embeds = np.asarray([r.embedding for r in results])
            similarities = cosine_sim(sent_embed, embeds)
        else:
            embeds = np.asarray([
                get_embedding(cosine_model, embed_cache, sent) for sent in
                [sentence, *[" ".join(res.content) for res in results]]
            ])
            similarities = cosine_sim(embeds[0], embeds[1:])

    processed_doc = []
    if prepend_examples:
        handle_results(tokenizer, processed_doc, results, use_labels,
                       use_mentions, use_labels, use_mentions)
        #processed_doc.append(tokenizer.sep_token)

    processed_doc.extend(get_input_sentence(tokenizer, doc, insert_prefix))

    if not prepend_examples:
        #processed_doc.append(tokenizer.sep_token)
        handle_results(tokenizer, processed_doc, results, use_labels,
                       use_mentions, use_labels, use_mentions)

    return processed_doc, [result.score for result in results], similarities


def tokenize_database_json_with_filter(
        tokenizer: PreTrainedTokenizer,
        file_name,
        type_file,
        database: Pipeline,
        cosine_model: Optional[SentenceTransformer],
        embed_cache,
        use_labels: bool,
        use_mentions: bool,
        output_path,
        filters: dict = {},
        filter_exact_match: bool = True,
        filter_same_document: bool = True,
        filtered_document_ids=[],
        prepend_task_description=True,
        prepend_search_results=False):
    if MENTION_START not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_START)
    if MENTION_END not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_END)

    with open(type_file, encoding="utf-8") as file:
        labels = json.load(file)['entities']
    label_to_id = {label: id for id, label in enumerate(labels)}

    tokenized_dataset = []
    with open(file_name, encoding="utf-8") as file:
        instances = json.load(file)

    name = os.path.basename(os.path.splitext(file_name)[0])
    database_scores = {}
    database_similarities = {}
    for inst_id, instance in tqdm(enumerate(instances),
                                  desc="Tokenization with DB",
                                  total=len(instances)):
        tokens = instance['tokens']
        entities = instance['entities']
        extended = instance['extended']
        doc_id = instance[
            'doc_id'] if "doc_id" in instance else name + "_" + str(inst_id)

        tokenized_sentence, target_sentence, entity_type_sequence, entity_indices, subtoken_map = get_target_sentence(
            tokenizer, label_to_id, tokens, entities)

        # insert prefix (instruction for model) here
        input_sentence, scores, similarities = get_input_sentence_database_with_filter(
            tokenizer,
            doc_id,
            extended,
            database,
            cosine_model,
            embed_cache,
            use_labels,
            use_mentions,
            filters,
            filter_exact_match=filter_exact_match,
            filter_same_document=filter_same_document,
            filtered_document_ids=filtered_document_ids,
            prepend_examples=prepend_search_results,
            insert_prefix=prepend_task_description)
        tokenized_dataset.append({
            "doc_id": doc_id,
            "sentence": tokenized_sentence,
            # sentence is for copy mechanism, might be different from
            # input_sentence which is for encoding only
            "input_sentence": input_sentence,
            "target_sentence": target_sentence,
            "subtoken_map": subtoken_map,
            "ent_type_sequence": entity_type_sequence,
            "ent_indices": entity_indices
        })
        database_scores[doc_id] = scores
        database_similarities[doc_id] = similarities

    output_file_name = os.path.join(
        output_path,
        f"{os.path.basename(os.path.splitext(file_name)[0])}.{os.path.basename(os.path.splitext(tokenizer.name_or_path)[0])}.jsonlines"
    )
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        json.dump(tokenized_dataset, output_file)
    return output_file_name, database_scores, database_similarities


def get_input_sentence_database(tokenizer: PreTrainedTokenizer,
                                doc,
                                search_result: List[Document],
                                sent_use_labels: bool,
                                sent_use_mentions: bool,
                                gaz_use_labels: bool,
                                gaz_use_mentions: bool,
                                prepend_examples=False,
                                insert_prefix=True):
    processed_doc = []
    if prepend_examples:
        handle_results(tokenizer, processed_doc, search_result,
                       sent_use_labels, sent_use_mentions, gaz_use_labels,
                       gaz_use_mentions)

    processed_doc.extend(get_input_sentence(tokenizer, doc, insert_prefix))

    if not prepend_examples:
        handle_results(tokenizer, processed_doc, search_result,
                       sent_use_labels, sent_use_mentions, gaz_use_labels,
                       gaz_use_mentions)

    return processed_doc


def query_database(instances: List, search: Pipeline):
    chunk_size = 1000
    with tqdm(desc="Querying database", total=len(instances)) as pbar:
        for i in range(0, len(instances), chunk_size):
            chunk = instances[i:i + chunk_size]
            results = search.run_batch(
                [" ".join(instance['extended']) for instance in chunk])
            for j, res in zip(
                    range(len(chunk)),
                    results["documents"] if results is not None else []):
                pbar.update(1)
                yield j + i, res


def tokenize_database_json(tokenizer: PreTrainedTokenizer,
                           file_name,
                           type_file,
                           search: Pipeline,
                           sent_use_labels: bool,
                           sent_use_mentions: bool,
                           gaz_use_labels: bool,
                           gaz_use_mentions: bool,
                           output_path,
                           prepend_task_description=True,
                           prepend_search_results=False):
    if MENTION_START not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_START)
    if MENTION_END not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_END)

    with open(type_file, encoding="utf-8") as file:
        labels = json.load(file)['entities']
    label_to_id = {label: id for id, label in enumerate(labels)}

    tokenized_dataset = []
    with open(file_name, encoding="utf-8") as file:
        instances = json.load(file)

    name = os.path.basename(os.path.splitext(file_name)[0])
    for instance_idx, search_result in tqdm(query_database(instances, search),
                                            desc="Tokenization with DB",
                                            total=len(instances)):
        instance = instances[instance_idx]
        tokens = instance['tokens']
        entities = instance['entities']
        extended = instance['extended']
        doc_id = instance[
            'doc_id'] if "doc_id" in instance else name + "_" + str(
                instance_idx)

        tokenized_sentence, target_sentence, entity_type_sequence, entity_indices, subtoken_map = get_target_sentence(
            tokenizer, label_to_id, tokens, entities)

        # insert prefix (instruction for model) here
        input_sentence = get_input_sentence_database(
            tokenizer,
            extended,
            search_result,
            sent_use_labels,
            sent_use_mentions,
            gaz_use_labels,
            gaz_use_mentions,
            prepend_examples=prepend_search_results,
            insert_prefix=prepend_task_description)
        tokenized_dataset.append({
            "doc_id": doc_id,
            "sentence": tokenized_sentence,
            # sentence is for copy mechanism, might be different from
            # input_sentence which is for encoding only
            "input_sentence": input_sentence,
            "target_sentence": target_sentence,
            "subtoken_map": subtoken_map,
            "ent_type_sequence": entity_type_sequence,
            "ent_indices": entity_indices
        })

    output_file_name = os.path.join(
        output_path,
        f"{os.path.basename(os.path.splitext(file_name)[0])}.{os.path.basename(os.path.splitext(tokenizer.name_or_path)[0])}.jsonlines"
    )
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        json.dump(tokenized_dataset, output_file)
    return output_file_name


def tokenize_search_results_json(tokenizer: PreTrainedTokenizer,
                                 file_name: str,
                                 type_file: str,
                                 search_results: Dict[int, List[Document]],
                                 output_path: str,
                                 output_name: Optional[str] = None,
                                 use_labels: Optional[bool] = None,
                                 use_mentions: Optional[bool] = None,
                                 sent_use_labels: Optional[bool] = None,
                                 sent_use_mentions: Optional[bool] = None,
                                 gaz_use_labels: Optional[bool] = None,
                                 gaz_use_mentions: Optional[bool] = None,
                                 prepend_task_description=True,
                                 prepend_search_results=False):
    if use_labels is not None and use_mentions is not None:
        assert sent_use_labels is sent_use_mentions is gaz_use_labels is gaz_use_mentions is None
        sent_use_labels = use_labels
        gaz_use_labels = use_labels
        sent_use_mentions = use_mentions
        gaz_use_mentions = use_mentions
    elif sent_use_labels is not None and sent_use_mentions is not None:
        assert use_labels is use_mentions is None
        if gaz_use_labels is None:
            gaz_use_labels = False
            gaz_use_mentions = False
    elif gaz_use_labels is not None and gaz_use_mentions is not None:
        assert use_labels is use_mentions is None
        if sent_use_labels is None:
            sent_use_labels = False
            sent_use_mentions = False
    else:
        use_labels = False
        use_mentions = False

    if MENTION_START not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_START)
    if MENTION_END not in tokenizer.get_vocab():
        tokenizer.add_tokens(MENTION_END)

    with open(type_file, encoding="utf-8") as file:
        labels = json.load(file)['entities']
    label_to_id = {label: id for id, label in enumerate(labels)}

    tokenized_dataset = []
    with open(file_name, encoding="utf-8") as file:
        instances = json.load(file)

    name = os.path.basename(os.path.splitext(file_name)[0])
    for instance_idx, instance in tqdm(enumerate(instances),
                                       desc="Tokenization with DB",
                                       total=len(instances)):
        tokens = instance['tokens']
        entities = instance['entities']
        extended = instance['extended']
        doc_id = instance[
            'doc_id'] if "doc_id" in instance else name + "_" + str(
                instance_idx)

        tokenized_sentence, target_sentence, entity_type_sequence, entity_indices, subtoken_map = get_target_sentence(
            tokenizer, label_to_id, tokens, entities)

        # insert prefix (instruction for model) here
        input_sentence = get_input_sentence_database(
            tokenizer,
            extended,
            search_results[instance_idx],
            sent_use_labels,  # type: ignore
            sent_use_mentions,  # type: ignore
            gaz_use_labels,  # type: ignore
            gaz_use_mentions,  # type: ignore
            prepend_examples=prepend_search_results,
            insert_prefix=prepend_task_description)
        tokenized_dataset.append({
            "doc_id": doc_id,
            "sentence": tokenized_sentence,
            # sentence is for copy mechanism, might be different from
            # input_sentence which is for encoding only
            "input_sentence": input_sentence,
            "target_sentence": target_sentence,
            "subtoken_map": subtoken_map,
            "ent_type_sequence": entity_type_sequence,
            "ent_indices": entity_indices
        })
    if output_name is None:
        output_file_name = os.path.join(
            output_path,
            f"{os.path.basename(os.path.splitext(file_name)[0])}.{os.path.basename(os.path.splitext(tokenizer.name_or_path)[0])}.jsonlines"
        )
    else:
        output_file_name = os.path.join(
            output_path,
            f"{output_name}.{os.path.basename(os.path.splitext(tokenizer.name_or_path)[0])}.jsonlines"
        )
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        json.dump(tokenized_dataset, output_file)
    return output_file_name
