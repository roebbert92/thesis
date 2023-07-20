from typing import Dict, Optional, List
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm import tqdm
from itertools import combinations_with_replacement
import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModel  # type: ignore


class SearchSampleSimilarity(torch.nn.Module):
    def __init__(self,
                 sbert_model_name: str,
                 inner_batch_size: int = 1000) -> None:
        super().__init__()
        self.inner_batch_size = inner_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)
        self.model = AutoModel.from_pretrained(sbert_model_name)
        self.cosine = torch.nn.CosineSimilarity(dim=-1)

    #Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, query_ids, queries, full_contexts, encoded_inputs):
        result = []
        encoded_inputs = encoded_inputs.to(self.model.device)
        with torch.no_grad():
            model_output = self.model(**encoded_inputs)
        embeds = SearchSampleSimilarity.mean_pooling(
            model_output, encoded_inputs["attention_mask"])

        for idx, q_id in enumerate(query_ids):
            res = {"doc_id": q_id}
            query_embed = embeds[queries[idx]].unsqueeze(0)
            try:
                full_start = full_contexts[idx][0]
                full_end = full_contexts[idx][1]
                full_size = full_end - full_start
                if full_size > 0:
                    full_contexts_embed = embeds[full_start:full_end]
                    full_contexts_cosine = self.cosine(
                        query_embed.repeat((full_size, 1)),
                        full_contexts_embed)
                    # calculate max + distribution
                    res.update({
                        "max": torch.max(full_contexts_cosine),
                    })
                else:
                    res.update({
                        "max": torch.nan,
                    })

            except:
                print(queries, full_contexts, sep="\n")
            result.append(res)

        return result


class SearchSampleDataset(Dataset):
    def __init__(self, dataset: List[dict], search_results: dict):
        self.search_results = search_results
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        results = [doc.to_dict() for doc in self.search_results[index]]
        _, full_contexts = get_full_context(results)
        return (sample["doc_id"], " ".join(sample["tokens"]), full_contexts)


class SearchSampleSimilarityCollator(object):
    def __init__(self, sbert_model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)

    def __call__(self, collator_input):
        query_ids = []
        queries = []
        full_contexts = []
        sentences = []
        for query_id, query, full in collator_input:
            query_ids.append(query_id)
            queries.append(len(sentences))
            sentences.append(query)
            full_start = len(sentences)
            sentences.extend(full)
            full_end = len(sentences)
            full_contexts.append((full_start, full_end))

        encoded_sentences = self.tokenizer(sentences,
                                           padding=True,
                                           truncation=True,
                                           max_length=128,
                                           return_tensors='pt')

        return query_ids, queries, full_contexts, encoded_sentences


def get_search_sample_similarity(dataset: List[dict], search_results: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-mpnet-base-v2" if device == "cuda" else "sentence-transformers/all-MiniLM-L6-v2"
    model = SearchSampleSimilarity(model_name).to(device)
    collator = SearchSampleSimilarityCollator(model_name)
    data = SearchSampleDataset(dataset, search_results)
    loader = DataLoader(data, batch_size=20, collate_fn=collator)

    for query_id, query, full_contexts, encoded_inputs in loader:
        for item in model(query_id, query, full_contexts, encoded_inputs):
            yield item


def batched_similarity(fn: torch.nn.Module, x_tensors: Tensor,
                       compare_tensors: Tensor, scores: Tensor, idx: int,
                       set_batch_size: int, device: str):
    if set_batch_size > 1:
        _sims = []
        for x_tensor, compare_tensor in zip(
                torch.tensor_split(x_tensors, set_batch_size, dim=1),
                torch.tensor_split(compare_tensors, set_batch_size, dim=1)):
            _sims.append(
                fn(x_tensor.to(device=device),
                   compare_tensor.to(device=device)))
        sims = torch.cat(_sims, dim=-1)
    else:
        sims = fn(x_tensors.to(device=device),
                  compare_tensors.to(device=device))
    best_scores = torch.max(sims, dim=-1).values
    len_scores = best_scores.shape[0]
    scores[idx - len_scores:idx] = best_scores.cpu()


def compute_similarity(fn: torch.nn.Module,
                       first_name: str,
                       first_embed: Tensor,
                       second_name: str,
                       second_embed: Tensor,
                       device: str,
                       batch_size: int = 10,
                       max_set_size: int = 20000):
    def compute(first_name: str, first: Tensor, second_name: str,
                second: Tensor, device: str, batch_size: int,
                max_set_size: int):
        embedding_dim = first.shape[1]
        set_size = len(second)
        set_batch_size = set_size // max_set_size + 1
        first_to_second = torch.empty((len(first), ))
        x_tensors = torch.empty((batch_size, set_size, embedding_dim))
        compare_tensors = torch.empty((batch_size, set_size, embedding_dim))
        # take best score
        batch_idx = 0
        for idx, x in tqdm(enumerate(first),
                           total=len(first),
                           desc=f"{second_name} to {first_name} "):
            batch_idx = idx % batch_size
            if idx > 0 and batch_idx == 0:
                batched_similarity(fn,
                                   x_tensors,
                                   compare_tensors,
                                   first_to_second,
                                   idx,
                                   set_batch_size,
                                   device=device)
            x_tensors[batch_idx] = x.repeat((set_size, 1))
            compare_tensors[batch_idx] = second

        if batch_idx > 0:
            batched_similarity(fn,
                               x_tensors[:batch_idx + 1],
                               compare_tensors[:batch_idx + 1],
                               first_to_second,
                               len(first),
                               set_batch_size,
                               device=device)
        return first_to_second

    if first_name == second_name:
        sims = compute(first_name, first_embed, second_name, second_embed,
                       device, batch_size,
                       max_set_size).cpu().numpy().tolist()
        yield "second_to_first", sims
        yield "first_to_second", sims
    else:
        yield "second_to_first", compute(first_name, first_embed, second_name,
                                         second_embed, device, batch_size,
                                         max_set_size).cpu().numpy().tolist()

        yield "first_to_second", compute(second_name, second_embed, first_name,
                                         first_embed, device, batch_size,
                                         max_set_size).cpu().numpy().tolist()


def get_embeddings(cache: Dict[str, Tensor], model: SentenceTransformer,
                   first_name: str, first: List[str], second_name: str,
                   second: List[str]):
    if first_name in cache:
        first_embed = cache[first_name]
    else:
        first_embed: Tensor = torch.stack([
            tensor.to(dtype=torch.bfloat16, device="cpu")
            for tensor in model.encode(first,
                                       convert_to_numpy=False,
                                       convert_to_tensor=False,
                                       show_progress_bar=True)
        ]).cpu()
        cache[first_name] = first_embed
    if second_name in cache:
        second_embed = cache[second_name]
    else:
        second_embed: Tensor = torch.stack([
            tensor.to(dtype=torch.bfloat16, device="cpu")
            for tensor in model.encode(second,
                                       convert_to_numpy=False,
                                       convert_to_tensor=False,
                                       show_progress_bar=True)
        ]).cpu()
        cache[second_name] = second_embed
    return first_embed, second_embed


def get_windowed_context(dataset: List[dict], window_size=3):
    contexts = set()

    def windowed_context(tokens: List[str], entities):
        for entity in entities:
            if entity["end"] - entity["start"] == 0:
                entity["end"] += 1
            entity_text = " ".join(tokens[entity["start"]:entity["end"]])
            if entity["start"] < window_size:
                entity["start"] = 0
            else:
                entity["start"] -= window_size
            if entity["end"] > len(tokens) - 1 - window_size:
                entity["end"] = len(tokens) - 1
            else:
                entity["end"] += window_size
            entity_context = " ".join(tokens[entity["start"]:entity["end"]])
            contexts.add((entity_context, entity_text, entity["type"]))

    for item in dataset:
        if "entities" in item:
            ents = copy.deepcopy(item["entities"])
            windowed_context(item["tokens"], ents)
        elif "meta" in item:
            # search result
            if item["meta"]["data_type"] == "gazetteers":
                # gazetteer - take full context
                contexts.add(
                    (item['content'], item["content"], item["meta"]["type"]))
            elif item["meta"]["data_type"] == "sentences":
                # sentence
                ents = copy.deepcopy(item["meta"]["entities"])
                windowed_context(item["content"].split(" "), ents)

    return ["_".join(e[1:]) for e in contexts], [e[0] for e in contexts]


def get_full_context(dataset: List[dict]):
    sentences = []
    sentence_ids = []
    for item_idx, item in enumerate(dataset):
        if "tokens" in item:
            sentence_ids.append(item["doc_id"])
            sentences.append(" ".join(item["tokens"]))
        elif "content" in item:
            # search result
            sentence_ids.append(item["id"])
            if item["meta"]["data_type"] == "gazetteers":
                sentences.append(f"{item['meta']['type']}: {item['content']}")
            else:
                sentences.append(item["content"])
        elif "entity" in item:
            sentence_ids.append(item_idx)
            sentences.append(f"{item['entity']}")
    return sentence_ids, sentences


def sample_similarity(first_name: str,
                      first: List[dict],
                      second_name: str,
                      second: List[dict],
                      cache: Dict[str, Tensor] = {}):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
        if device == "cuda" else "sentence-transformers/all-MiniLM-L6-v2",
        device=device)
    cosine = torch.nn.CosineSimilarity(dim=-1)

    # build database (gazetteers (entities), sentences) if not exists for each path
    #first_gaz_ids, first_gazetteers = get_windowed_context(first)
    first_sent_ids, first_sentences = get_full_context(first)
    #second_gaz_ids, second_gazetteers = get_windowed_context(second)
    second_sent_ids, second_sentences = get_full_context(second)

    with torch.no_grad():
        # first_gaz_embed, second_gaz_embed = get_embeddings(
        #     cache, model, first_name + "_gaz", first_gazetteers,
        #     second_name + "_gaz", second_gazetteers, device)
        # for direction, gaz_sims in compute_similarity(
        #         cosine, first_name + "_gaz", first_gaz_embed,
        #         second_name + "_gaz", second_gaz_embed, device):
        #     if direction == "first_to_second":
        #         yield "first", "windowed", second_gaz_ids, gaz_sims
        #     if direction == "second_to_first":
        #         yield "second", "windowed", first_gaz_ids, gaz_sims
        # del first_gaz_embed
        # del second_gaz_embed

        first_sent_embed, second_sent_embed = get_embeddings(
            cache, model, first_name + "_sent", first_sentences,
            second_name + "_sent", second_sentences)
        for direction, sent_sims in compute_similarity(
                cosine, first_name + "_sent", first_sent_embed,
                second_name + "_sent", second_sent_embed, device):
            if direction == "first_to_second":
                yield "first", "full", second_sent_ids, sent_sims
            if direction == "second_to_first":
                yield "second", "full", first_sent_ids, sent_sims
        del first_sent_embed
        del second_sent_embed

        if device == "cuda":
            torch.cuda.empty_cache()


def display_cases_sample_similarity(cm_ss: pd.DataFrame,
                                    first_name: str,
                                    second_name: str,
                                    context_type: str,
                                    ax: Optional[plt.Axes] = None):
    df = cm_ss.loc[(cm_ss["first"] == first_name)
                   & (cm_ss["second"] == second_name) &
                   (cm_ss["context_type"] == context_type)]
    df = df.groupby("data_id").mean()
    case1 = df[(0.5 < df.cosine_similarity) & (df.cosine_similarity <= 1.0)]
    case2 = df[(0.0 < df.cosine_similarity) & (df.cosine_similarity <= 0.5)]
    case3 = df[(-0.5 < df.cosine_similarity) & (df.cosine_similarity <= 0.0)]
    case4 = df[(-1 < df.cosine_similarity) & (df.cosine_similarity <= -0.5)]

    x = ["φ ∈ (0.5,1]", "φ ∈ (0,0.5]", "φ ∈ (-0.5,0]", "φ ∈ [-1,-0.5]"]
    x_ticks = np.arange(len(x))
    y = [len(case1), len(case2), len(case3), len(case4)]

    if ax is None:
        bar_container = plt.bar(x_ticks, y, align="center")
    else:
        bar_container = ax.bar(x_ticks, y, align="center")
    plt.xticks(x_ticks, x)
    plt.ylabel(
        f"# of {'entities' if context_type == 'windowed' else 'sentences'}")
    plt.bar_label(bar_container, y)
    plt.title(f"Context Similarity Histogram {first_name} to {second_name}")

    return plt.figure() if ax is None else ax.figure


def confusion_matrix_sample_similarity(datasets: List[List[dict]],
                                       names: List[str]):
    similarities = []
    cache = {}
    for (left, left_dataset), (right,
                               right_dataset) in combinations_with_replacement(
                                   zip(names, datasets), 2):
        similarity = sample_similarity(left,
                                       left_dataset,
                                       right,
                                       right_dataset,
                                       cache=cache)
        for direction, context_type, data_ids, data in similarity:
            first = left if direction == "first" else right
            second = right if direction == "first" else left
            similarities.extend([{
                "first": first,
                "second": second,
                "context_type": context_type,
                "data_id": id,
                "cosine_similarity": d
            } for id, d in zip(data_ids, data)])

    df = pd.DataFrame.from_records(similarities)
    df.cosine_similarity.clip(-1.0, 1.0, inplace=True)
    return df