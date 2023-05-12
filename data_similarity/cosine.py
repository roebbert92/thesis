from typing import Dict, Optional, List, Tuple
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor
from tqdm import tqdm


def batched_similarity(fn: torch.nn.Module, x_tensors: Tensor,
                       compare_tensors: Tensor, scores: Tensor, idx: int,
                       set_batch_size: int):
    if set_batch_size > 1:
        _sims = []
        for x_tensor, compare_tensor in zip(
                torch.tensor_split(x_tensors, set_batch_size, dim=1),
                torch.tensor_split(compare_tensors, set_batch_size, dim=1)):
            _sims.append(fn(x_tensor, compare_tensor))
        sims = torch.cat(_sims, dim=-1)
    else:
        sims = fn(x_tensors, compare_tensors)
    best_scores = torch.max(sims, dim=-1).values
    len_scores = best_scores.shape[0]
    scores[idx - len_scores:idx] = best_scores


def compute_similarity(fn: torch.nn.Module,
                       first_embed: Tensor,
                       second_embed: Tensor,
                       is_same_dataset: bool,
                       device: str,
                       batch_size: int = 20,
                       max_set_size: int = 3000):
    embedding_dim = first_embed.shape[1]
    set_size = len(second_embed) - 1 if is_same_dataset else len(second_embed)
    set_batch_size = set_size // max_set_size + 1
    first_to_second = torch.empty((len(first_embed), ), device=device)
    x_tensors = torch.empty((batch_size, set_size, embedding_dim),
                            device=device)
    compare_tensors = torch.empty((batch_size, set_size, embedding_dim),
                                  device=device)
    # take best score
    batch_idx = 0
    for idx, x in tqdm(enumerate(first_embed),
                       total=len(first_embed),
                       desc="first"):
        batch_idx = idx % batch_size
        if idx > 0 and batch_idx == 0:
            batched_similarity(fn, x_tensors, compare_tensors, first_to_second,
                               idx, set_batch_size)
        x_tensors[batch_idx] = x.repeat((set_size, 1))
        if is_same_dataset:
            indices = torch.tensor(
                [i for i in range(len(first_embed)) if i != idx],
                device=device)
            compare_tensors[batch_idx] = torch.index_select(
                second_embed, 0, indices)
        else:
            compare_tensors[batch_idx] = second_embed

    if batch_idx > 0:
        batched_similarity(fn, x_tensors[:batch_idx + 1],
                           compare_tensors[:batch_idx + 1], first_to_second,
                           len(first_embed), set_batch_size)

    yield "first_to_second", first_to_second.cpu().numpy().tolist()

    if is_same_dataset:
        second_to_first = first_to_second
    else:
        second_to_first = torch.empty((len(second_embed), ), device=device)
        x_tensors = torch.empty((batch_size, len(first_embed), embedding_dim),
                                device=device)
        compare_tensors = torch.empty(
            (batch_size, len(first_embed), embedding_dim), device=device)
        set_batch_size = len(first_embed) // max_set_size + 1
        batch_idx = 0
        for idx, x in tqdm(enumerate(second_embed),
                           total=len(second_embed),
                           desc="second"):
            batch_idx = idx % batch_size
            if idx > 0 and batch_idx == 0:
                batched_similarity(fn, x_tensors, compare_tensors,
                                   second_to_first, idx, set_batch_size)
            x_tensors[batch_idx] = x.repeat((len(first_embed), 1))
            compare_tensors[batch_idx] = first_embed

        if batch_idx > 0:
            batched_similarity(fn, x_tensors[:batch_idx + 1],
                               compare_tensors[:batch_idx + 1],
                               second_to_first, len(second_embed),
                               set_batch_size)

    # avg
    # avg_first_to_second = torch.mean(first_to_second).cpu().numpy()
    # avg_second_to_first = torch.mean(second_to_first).cpu().numpy()

    yield "second_to_first", second_to_first.cpu().numpy().tolist()


def get_embeddings(cache: Dict[str, Tensor], model: SentenceTransformer,
                   first_name: str, first: List[str], second_name: str,
                   second: List[str], is_same_dataset: bool, device: str):
    if first_name in cache:
        first_embed = cache[first_name].to(device=device)
    else:
        first_embed: Tensor = model.encode(
            first, convert_to_numpy=False,
            convert_to_tensor=True)  # type: ignore
        cache[first_name] = first_embed.cpu()
    if is_same_dataset:
        second_embed = first_embed
    else:
        if second_name in cache:
            second_embed = cache[second_name].to(device=device)
        else:
            second_embed: Tensor = model.encode(
                second, convert_to_numpy=False,
                convert_to_tensor=True)  # type: ignore
            cache[second_name] = second_embed.cpu()
    return first_embed, second_embed


def get_gazetteers(dataset: List[dict]):
    entities = set()
    for item in dataset:
        for entity in item["entities"]:
            if entity["end"] - entity["start"] == 0:
                entity["end"] += 1
            entities.add(
                (" ".join(item["tokens"][entity["start"]:entity["end"]]),
                 entity["type"]))
    return ["_".join(e) for e in entities], [e[0] for e in entities]


def get_sentences(dataset: List[dict]):
    sentences = []
    sentence_ids = []
    for item in dataset:
        sentence_ids.append(item["doc_id"])
        sentences.append(" ".join(item["tokens"]))
    return sentence_ids, sentences


def dataset_similarity(cache: Dict[str, Tensor],
                       first_name: str,
                       first: List[dict],
                       second_name: Optional[str] = None,
                       second: Optional[List[dict]] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
        if device == "cuda" else "sentence-transformers/all-MiniLM-L6-v2",
        device=device)
    cosine = torch.nn.CosineSimilarity(dim=-1)

    is_same_dataset = second is None
    # build database (gazetteers (entities), sentences) if not exists for each path
    first_gaz_ids, first_gazetteers = get_gazetteers(first)
    first_sent_ids, first_sentences = get_sentences(first)
    if is_same_dataset:
        second_gaz_ids, second_gazetteers = first_gaz_ids, first_gazetteers
        second_sent_ids, second_sentences = first_sent_ids, first_sentences
        second_name = first_name
    else:
        second_gaz_ids, second_gazetteers = get_gazetteers(second)
        second_sent_ids, second_sentences = get_sentences(second)

    # if both paths are the same, filter out same doc.id
    # Optional:
    # for gazetteers: 0.5*cosine(content) + 0.5 * KL(1 if same type, 0 if other type)
    # for sentences: 0.5*cosine(content) + 0.5 * Avg(Gazetteers)
    with torch.no_grad():
        first_gaz_embed, second_gaz_embed = get_embeddings(
            cache, model, first_name + "_gaz", first_gazetteers,
            second_name + "_gaz", second_gazetteers, is_same_dataset, device)
        for direction, gaz_sims in compute_similarity(cosine, first_gaz_embed,
                                                      second_gaz_embed,
                                                      is_same_dataset, device):
            if direction == "first_to_second":
                yield "first", "gazetteers", first_gaz_ids, gaz_sims
            if direction == "second_to_first":
                yield "second", "gazetteers", second_gaz_ids, gaz_sims
        del first_gaz_embed
        del second_gaz_embed

        first_sent_embed, second_sent_embed = get_embeddings(
            cache, model, first_name + "_sent", first_sentences,
            second_name + "_sent", second_sentences, is_same_dataset, device)
        for direction, sent_sims in compute_similarity(cosine,
                                                       first_sent_embed,
                                                       second_sent_embed,
                                                       is_same_dataset,
                                                       device):
            if direction == "first_to_second":
                yield "first", "sentences", first_sent_ids, sent_sims
            if direction == "second_to_first":
                yield "second", "sentences", second_sent_ids, sent_sims
        del first_sent_embed
        del second_sent_embed

        if device == "cuda":
            torch.cuda.empty_cache()
