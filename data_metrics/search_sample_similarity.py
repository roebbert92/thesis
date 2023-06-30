from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel  # type: ignore
import sys
import os
from tqdm import tqdm
import pandas as pd

thesis_path = "/" + os.path.join(
    *os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])
sys.path.append(thesis_path)

from data_metrics.sample_similarity import get_full_context


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

    def forward(self, query_ids, queries, full_contexts, entity_counts,
                entity_count_per_samples, encoded_inputs):
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
                total_entity_count = entity_counts[idx]
                if full_size > 0:
                    full_contexts_embed = embeds[full_start:full_end]
                    full_contexts_cosine = self.cosine(
                        query_embed.repeat((full_size, 1)),
                        full_contexts_embed)
                    total_entity_count = entity_counts[idx]
                    search_score = torch.sum(
                        torch.tensor(entity_count_per_samples[idx]).to(
                            self.model.device) * full_contexts_cosine
                    )
                    # calculate max + distribution
                    res.update({
                        "total entities":
                        total_entity_count,
                        "search score":
                        search_score.item(),
                        "max":
                        torch.max(full_contexts_cosine).item(),
                        "φ ∈ (0.5,1]":
                        torch.sum(
                            torch.logical_and(
                                0.5 < full_contexts_cosine,
                                full_contexts_cosine <= 1.0)).item(),
                        "φ ∈ (0,0.5]":
                        torch.sum(
                            torch.logical_and(
                                0.0 < full_contexts_cosine,
                                full_contexts_cosine <= 0.5)).item(),
                        "φ ∈ (-0.5,0]":
                        torch.sum(
                            torch.logical_and(
                                -0.5 < full_contexts_cosine,
                                full_contexts_cosine <= 0.0)).item(),
                        "φ ∈ [-1,-0.5]":
                        torch.sum(
                            torch.logical_and(
                                -1.0 <= full_contexts_cosine,
                                full_contexts_cosine <= -0.5)).item(),
                    })
                else:
                    res.update({
                        "total entities": total_entity_count,
                        "search score": torch.nan,
                        "max": torch.nan,
                        "φ ∈ (0.5,1]": torch.tensor(0).item(),
                        "φ ∈ (0,0.5]": torch.tensor(0).item(),
                        "φ ∈ (-0.5,0]": torch.tensor(0).item(),
                        "φ ∈ [-1,-0.5]": torch.tensor(0).item(),
                    })

            except:
                print(queries, full_contexts, sep="\n")
            result.append(res)

        return result


def get_entity_counts(dataset: List[dict]):
    total_entity_count = 0
    entity_count_per_sample = []
    for item in dataset:
        if "tokens" in item:
            entity_count = len(item["entities"])
            total_entity_count += entity_count
            entity_count_per_sample.append(entity_count)
        elif "content" in item:
            # search result
            if item["meta"]["data_type"] == "gazetteers":
                entity_count = 1
                total_entity_count += entity_count
                entity_count_per_sample.append(entity_count)
            else:
                entity_count = len(item["meta"]["entities"])
                total_entity_count += entity_count
                entity_count_per_sample.append(entity_count)
    return total_entity_count, entity_count_per_sample


class SearchSampleDataset(Dataset):
    def __init__(self, dataset: List[dict], search_results: dict):
        self.search_results = search_results
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        results = self.search_results[index]
        _, full_contexts = get_full_context(results)
        entity_count, entity_count_per_sample = get_entity_counts(results)
        return (sample["doc_id"], " ".join(sample["tokens"]), full_contexts,
                entity_count, entity_count_per_sample)


class SearchSampleSimilarityCollator(object):
    def __init__(self, sbert_model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)

    def __call__(self, collator_input):
        query_ids = []
        queries = []
        full_contexts = []
        sentences = []
        entity_counts = []
        entity_count_per_samples = []
        for query_id, query, full, entity_count, entity_count_per_sample in collator_input:
            query_ids.append(query_id)
            queries.append(len(sentences))
            sentences.append(query)
            full_start = len(sentences)
            sentences.extend(full)
            full_end = len(sentences)
            full_contexts.append((full_start, full_end))
            entity_counts.append(entity_count)
            entity_count_per_samples.append(entity_count_per_sample)

        encoded_sentences = self.tokenizer(sentences,
                                           padding=True,
                                           truncation=True,
                                           max_length=128,
                                           return_tensors='pt')

        return query_ids, queries, full_contexts, entity_counts, entity_count_per_samples, encoded_sentences


def get_search_sample_similarity(dataset: List[dict], search_results: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-mpnet-base-v2" if device == "cuda" else "sentence-transformers/all-MiniLM-L6-v2"
    model = SearchSampleSimilarity(model_name).to(device)
    collator = SearchSampleSimilarityCollator(model_name)
    data = SearchSampleDataset(dataset, search_results)
    loader = DataLoader(data, batch_size=20, collate_fn=collator)

    for query_id, query, full_contexts, entity_counts, entity_count_per_samples, encoded_inputs in tqdm(
            loader, desc="Search Sample Similarity", position=1):
        for item in model(query_id, query, full_contexts, entity_counts,
                          entity_count_per_samples, encoded_inputs):
            yield item