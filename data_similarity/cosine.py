from typing import Optional, List
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor
import copy


def compute_similarity(fn: torch.nn.Module, first_embed: List[Tensor],
                       second_embed: List[Tensor], is_same_dataset: bool):
    first_to_second = []
    second_to_first = []
    # take best score
    for idx, x in enumerate(first_embed):
        compare = copy.deepcopy(second_embed)
        if is_same_dataset:
            del compare[idx]
        x_tensor = x.repeat((len(compare), 1))
        compare_tensor = torch.stack(compare)
        sims = fn(x_tensor, compare_tensor)
        best_score = torch.max(sims).cpu().numpy()
        first_to_second.append(float(best_score))

    if is_same_dataset:
        second_to_first = first_to_second
    else:
        for idx, x in enumerate(second_embed):
            x_tensor = x.repeat((len(first_embed), 1))
            compare_tensor = torch.stack(first_embed)
            sims = fn(x_tensor, compare_tensor)
            best_score = torch.max(sims).cpu().numpy()
            second_to_first.append(float(best_score))

    # avg
    avg_first_to_second = torch.mean(torch.tensor(first_to_second)).numpy()
    avg_second_to_first = torch.mean(torch.tensor(second_to_first)).numpy()

    return float(avg_first_to_second), float(avg_second_to_first)


def get_embeddings(model: SentenceTransformer, first: List[str],
                   second: List[str], is_same_dataset: bool):
    first_embed: List[Tensor] = model.encode(
        first, convert_to_numpy=False)  # type: ignore
    if is_same_dataset:
        second_embed = first_embed
    else:
        second_embed: List[Tensor] = model.encode(
            second, convert_to_numpy=False)  # type: ignore
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
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu")
    cosine = torch.nn.CosineSimilarity()

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
    first_gaz_embed, second_gaz_embed = get_embeddings(model, first_gazetteers,
                                                       second_gazetteers,
                                                       is_same_dataset)
    gazetteer_sim = compute_similarity(cosine, first_gaz_embed,
                                       second_gaz_embed, is_same_dataset)

    first_sent_embed, second_sent_embed = get_embeddings(
        model, first_sentences, second_sentences, is_same_dataset)
    sentence_sim = compute_similarity(cosine, first_sent_embed,
                                      second_sent_embed, is_same_dataset)

    # repeat for other side if not the same path
    return {"gazetteers": gazetteer_sim, "sentences": sentence_sim}