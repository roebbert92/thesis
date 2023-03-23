import math
from typing import Optional, Union, Iterable
import torch
from transformers.activations import ACT2FN
import torch.nn as nn
import torch.nn.functional as F

def logsumexp(
    tensor: torch.Tensor,
    dim: Union[int, Iterable] = -1,
    keepdim: bool = False
) -> torch.Tensor:
    if type(dim) == int:
        if tensor.size(dim) == 0:
            return tensor.sum(dim=dim, keepdim=keepdim).log()  # neginf
    else:
        for d in dim:
            assert type(dim) == int
            if tensor.size(d) == 0:
                return tensor.sum(dim=dim, keepdim=keepdim).log()  # neginf

    max_score = tensor.amax(dim, keepdim=True)
    stable_vec = tensor - max_score

    return max_score.sum(dim=dim, keepdim=keepdim) +\
        stable_vec.logsumexp(dim=dim, keepdim=keepdim)


def batched_masked_select(tensor: torch.FloatTensor, mask: torch.Tensor):
    max_len = mask.sum(dim=-1).max()  # maximum number of selected elements
    mask_sorted, indices = torch.sort(
        mask.long(), descending=True, stable=True, dim=-1
    )
    mask_sorted = mask_sorted.bool()

    return dim_batched_index_select(tensor, indices)[:, :max_len], mask_sorted[:, :max_len]


def get_range_vector(size: int, device: int) -> torch.Tensor:
    '''
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    '''
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    # Shape: (batch_size)
    offsets = get_range_vector(indices.size(
        0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def dim_batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    dim: int = 1,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # Input:
    #   target: (batch_size, seq_len, dim)
    #   indices: (batch_size, num_indices)
    # Returns:
    #   (batch_size, num_indices, dim)
    unidim = False
    if len(target.size()) == len(indices.size()):
        # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        unidim = True
        target = target.unsqueeze(-1)

    target_size = target.size()
    indices_size = indices.size()

    target = target.reshape(
        math.prod([*target_size[:dim]]), *target_size[dim:])
    indices = indices.view(
        math.prod([*indices_size[:dim]]), *indices_size[dim:])

    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(
            indices, target.size(1)
        )

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, *target_size[dim+1:])
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices_size) if unidim else (
        list(indices_size) + list(target_size[dim+1:]))
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.reshape(*selected_shape)
    return selected_targets

def make_linear(in_features, out_features, bias=True, std=0.02):
    # putting Linear on the last device so it's the same with t5 last_hidden_state
    device = torch.cuda.device_count() - 1 if torch.cuda.is_available() else "cpu"
    linear = nn.Linear(
        in_features, out_features, bias,
        device=device
    )
    linear.weight.data.normal_(mean=0.0, std=std)
    if bias:
        linear.bias.data.zero_()
    return linear

def make_ffnn(
    feat_size, hidden_size, output_size, dropout, std=0.02, activation='relu'
):
    if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
        return make_linear(feat_size, output_size, std=std)
    if not isinstance(hidden_size, Iterable):
        hidden_size = [hidden_size]

    ffnn = [make_linear(feat_size, hidden_size[0], std=std), ACT2FN[activation], dropout]
    for i in range(1, len(hidden_size)):
        ffnn += [make_linear(hidden_size[i-1], hidden_size[i], std=std), ACT2FN[activation], dropout]

    ffnn += [make_linear(hidden_size[-1], output_size, std=std)]
    return nn.Sequential(*ffnn)

def one_hot_ignore_negative(labels, num_classes):
    return F.one_hot(
        torch.where((labels >= 0), labels, num_classes),
        num_classes=num_classes+1
    )[..., :-1].bool()

def prepare_pair_embeddings(col_vecs: torch.FloatTensor, row_vecs: torch.FloatTensor):
    # Params: col_vecs: (num_col_vecs, dim_col_vec)
    #         row_vecs:    (num_row_vecs, dim_row_vec)
    # Returns : 
    # [[col_vec0:row_vec0, col_vec1:row_vec0, col_vec2:row_vec0, ...], 
    #  [col_vec0:row_vec1, col_vec1:row_vec1, col_vec2:row_vec1, ...], ...]
    # (num_row_vecs, num_col_vecs, dim_col_vec+dim_row_vec)
    if len(row_vecs.size()) == 2: # no beam size
        return torch.cat(
            [col_vecs.unsqueeze(0).expand(row_vecs.size(0), -1, -1), 
             row_vecs.unsqueeze(1).expand(-1, col_vecs.size(0), -1)], dim=-1
        )
    else:
        return torch.cat(
            [col_vecs.unsqueeze(1).expand(-1, row_vecs.size(1), -1, -1), 
             row_vecs.unsqueeze(2).expand(-1, -1, col_vecs.size(1), -1)], dim=-1
        )