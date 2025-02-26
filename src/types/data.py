from typing import List, TypeVar, TypedDict

import torch

T = TypeVar('T')

t_dataset_splits = TypedDict(
    't_dataset_splits',
    {'train': List[int], 'val': List[int], 'test': List[int]},
)

t_dataset_item = TypedDict('t_dataset_item', {
    "features": torch.Tensor,
    "profile": torch.Tensor,
    "control": torch.Tensor,
    "peak_id": int,
    "peak_start": int,
    "peak_end": int,
    "start": int,
    "end": int,
    "chr": str,
})

t_dataset_item_wo_control = TypedDict('t_dataset_item_wo_control', {
    "features": torch.Tensor,
    "profile": torch.Tensor,
    "peak_id": int,
    "peak_start": int,
    "peak_end": int,
    "start": int,
    "end": int,
    "chr": str,
})
