from typing import Optional, List
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor
from tqdm import tqdm


def batched_similarity(fn: torch.nn.Module, x_tensors: Tensor,
                       compare_tensors: Tensor, scores: Tensor, idx: int):
    sims = fn(x_tensors, compare_tensors)
    best_scores = torch.max(sims, dim=-1).values
    len_scores = best_scores.shape[0]
    scores[idx - len_scores:idx] = best_scores


def compute_similarity(fn: torch.nn.Module,
                       first_embed: Tensor,
                       second_embed: Tensor,
                       is_same_dataset: bool,
                       device: str,
                       batch_size: int = 20):
    embedding_dim = first_embed.shape[1]
    set_size = len(second_embed) - 1 if is_same_dataset else len(second_embed)
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
                               idx)
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
                           len(first_embed))

    if is_same_dataset:
        second_to_first = first_to_second
    else:
        second_to_first = torch.empty((len(second_embed), ), device=device)
        x_tensors = torch.empty((batch_size, len(first_embed), embedding_dim),
                                device=device)
        compare_tensors = torch.empty(
            (batch_size, len(first_embed), embedding_dim), device=device)
        batch_idx = 0
        for idx, x in tqdm(enumerate(second_embed),
                           total=len(second_embed),
                           desc="second"):
            batch_idx = idx % batch_size
            if idx > 0 and batch_idx == 0:
                batched_similarity(fn, x_tensors, compare_tensors,
                                   second_to_first, idx)
            x_tensors[batch_idx] = x.repeat((len(first_embed), 1))
            compare_tensors[batch_idx] = first_embed

        if batch_idx > 0:
            batched_similarity(fn, x_tensors[:batch_idx + 1],
                               compare_tensors[:batch_idx + 1],
                               second_to_first, len(second_embed))

    # avg
    avg_first_to_second = torch.mean(first_to_second).cpu().numpy()
    avg_second_to_first = torch.mean(second_to_first).cpu().numpy()

    return float(avg_first_to_second), float(avg_second_to_first)


def get_embeddings(model: SentenceTransformer, first: List[str],
                   second: List[str], is_same_dataset: bool):
    first_embed: Tensor = model.encode(first,
                                       convert_to_numpy=False,
                                       convert_to_tensor=True)  # type: ignore
    if is_same_dataset:
        second_embed = first_embed
    else:
        second_embed: Tensor = model.encode(
            second, convert_to_numpy=False,
            convert_to_tensor=True)  # type: ignore
    return first_embed, second_embed


def get_gazetteers(dataset: List[dict]) -> List[str]:
    entities = set()
    for item in dataset:
        for entity in item["entities"]:
            if entity["end"] - entity["start"] == 0:
                entity["end"] += 1
            entities.add(
                (" ".join(item["tokens"][entity["start"]:entity["end"]]),
                 entity["type"]))
    return [e[0] for e in entities]


def get_sentences(dataset: List[dict]) -> List[str]:
    sentences = []
    for item in dataset:
        sentences.append(" ".join(item["tokens"]))
    return sentences


def dataset_similarity(first: List[dict], second: Optional[List[dict]] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                device=device)
    cosine = torch.nn.CosineSimilarity(dim=-1)

    is_same_dataset = second is None
    # build database (gazetteers (entities), sentences) if not exists for each path
    first_gazetteers = get_gazetteers(first)
    first_sentences = get_sentences(first)
    if is_same_dataset:
        second_gazetteers = first_gazetteers
        second_sentences = first_sentences
    else:
        second_gazetteers = get_gazetteers(second)
        second_sentences = get_sentences(second)

    # if both paths are the same, filter out same doc.id
    # Optional:
    # for gazetteers: 0.5*cosine(content) + 0.5 * KL(1 if same type, 0 if other type)
    # for sentences: 0.5*cosine(content) + 0.5 * Avg(Gazetteers)
    with torch.no_grad():
        first_gaz_embed, second_gaz_embed = get_embeddings(
            model, first_gazetteers, second_gazetteers, is_same_dataset)
        gazetteer_sim = compute_similarity(cosine, first_gaz_embed,
                                           second_gaz_embed, is_same_dataset,
                                           device)
        del first_gaz_embed
        del second_gaz_embed

        first_sent_embed, second_sent_embed = get_embeddings(
            model, first_sentences, second_sentences, is_same_dataset)
        sentence_sim = compute_similarity(cosine, first_sent_embed,
                                          second_sent_embed, is_same_dataset,
                                          device)
        del first_sent_embed
        del second_sent_embed

        if device == "cuda":
            torch.cuda.empty_cache()

    # repeat for other side if not the same path
    return {"gazetteers": gazetteer_sim, "sentences": sentence_sim}