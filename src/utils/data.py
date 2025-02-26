import os
import os.path as osp
import random
from typing import List, cast

import numpy as np
import pandas as pd
import pyBigWig
import torch

from src.types import t_dataset_splits

tokenizer: dict[str, int] = {
    "A": 0, "C": 1, "G": 2, "T": 3,
    "N": 4, "R": 4, "Y": 4, "S": 4, "W": 4, "K": 4,
    "M": 4, "B": 4, "D": 4, "H": 4, "V": 4, "U": 4,
}


def replace_nans(np_array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np_array, nan=0)


def calc_avg_signal(
        chr_name: str,
        start: int,
        end: int,
        bws: List[pyBigWig.pyBigWig],
) -> np.ndarray:
    signals = [replace_nans(bw.values(chr_name, start, end, numpy=True)) for bw in bws]
    return np.mean(signals, axis=0)


def get_split_indices(
        split_dir: str,
        bed: pd.DataFrame,
        held_out_chrs: List[str],
        train_ratio: float,
        context_len: int,
        chr_lengths: dict[str, int],
        seed: int = 0,
) -> t_dataset_splits:
    split_path = osp.join(
        split_dir,
        f'held_out_chrs={",".join(held_out_chrs)}-train_ratio={train_ratio}-context_len={context_len}-seed={seed}.npz',
    )

    if osp.exists(split_path):
        print(f"Loading split from {split_path}")
        split_res = np.load(split_path, allow_pickle=True)
        return {
            'train': split_res['train'].tolist(),
            'val': split_res['val'].tolist(),
            'test': split_res['test'].tolist(),
        }

    print(f"Generating splits and saving to {split_path}")
    np.random.seed(seed)
    random.seed(seed)

    held_out_chrs = set(held_out_chrs)
    train_val_indices: List[int] = cast(List[int], [])
    test_indices: List[int] = cast(List[int], [])
    excluded_indices: List[int] = cast(List[int], [])
    for i, row in bed.iterrows():
        start = row['start']
        end = row['end']
        mid = (start + end) // 2
        chr_length = chr_lengths[cast(str, row['chr'])]
        if mid - context_len // 2 < 0 or mid + context_len // 2 > chr_length:
            print(f"Skipping {i}th data {row['chr']}:{start}-{end} due to {end} + {context_len // 2} > {chr_length}")
            excluded_indices.append(cast(int, i))
            continue
        if row['chr'] in held_out_chrs:
            test_indices.append(cast(int, i))
        else:
            train_val_indices.append(cast(int, i))

    n_train_val = len(train_val_indices)
    n_train = int(n_train_val * train_ratio)
    train_indices = sorted(np.random.choice(train_val_indices, n_train, replace=False).tolist())
    val_indices = sorted(list(set(train_val_indices) - set(train_indices)))

    # Sanity check
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    excluded_set = set(excluded_indices)
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    assert len(train_set & excluded_set) == 0
    assert len(val_set & excluded_set) == 0
    assert len(test_set & excluded_set) == 0
    assert train_set | val_set | test_set | excluded_set == set(range(len(bed)))

    os.makedirs(split_dir, exist_ok=True)
    np.savez_compressed(split_path, train=train_indices, val=val_indices, test=test_indices)

    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }


def adjust_dim(
        x: torch.Tensor,
        expected_channels: int
) -> torch.Tensor:
    if x.dim() == 3 and x.size(1) == expected_channels:
        return x

    if x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim == 3 and x.size(1) != expected_channels and x.size(2) == expected_channels:
        x = x.transpose(-1, -2)
    else:
        raise ValueError(f"Invalid input shape, expected channels: {expected_channels}, got: {x.shape}")
    return x
