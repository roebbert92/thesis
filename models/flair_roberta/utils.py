import random
import torch
import numpy as np
from lightning import seed_everything
from typing import List, Tuple


def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_to_max_length(
        batch: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
        max_len: int = None,
        fill_values: List[float] = None) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    doc_keys = [item[0] for item in batch]
    tensors = [item[1:] for item in batch]
    lengths = np.array([[len(field_data) for field_data in sample]
                        for sample in tensors])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = []
    output.append(doc_keys)
    output.extend([
        torch.full([batch_size, max_lengths[field_idx]],
                   fill_value=fill_values[field_idx],
                   dtype=tensors[0][field_idx].dtype)
        for field_idx in range(num_fields)
    ])
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = tensors[sample_idx][field_idx]
            output[field_idx + 1][sample_idx][:data.shape[0]] = data
    return output
