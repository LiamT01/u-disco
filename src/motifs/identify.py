import os
import pathlib

import modiscolite
import numpy as np
from modiscolite.util import calculate_window_offsets


def read_numpy_file(file_path: str) -> np.ndarray:
    assert file_path.endswith('.npy') or file_path.endswith('.npz'), \
        "File must be a .npy or .npz file"
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        return np.load(file_path)['arr_0']


def identify_motifs(
        ohe_seq_path: str,
        attr_path: str,
        output_dir: str,
        window_size: int,
        max_seqlets: int,
        n_leiden: int,
        verbose: bool,
):
    sequences = read_numpy_file(ohe_seq_path)
    attributions = read_numpy_file(attr_path)

    center = sequences.shape[2] // 2
    start, end = calculate_window_offsets(center, window_size)

    sequences = sequences[:, :, start:end].transpose(0, 2, 1)
    attributions = attributions[:, :, start:end].transpose(0, 2, 1)

    if sequences.shape[1] < window_size:
        raise ValueError(f"Window ({window_size}) cannot be larger than the sequence length (shape: {sequences.shape})")

    sequences = sequences.astype(np.float32)
    attributions = attributions.astype(np.float32)

    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=attributions,
        one_hot=sequences,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=20,
        flank_size=5,
        target_seqlet_fdr=0.05,
        n_leiden_runs=n_leiden,
        verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving identified motifs to {output_dir}", flush=True)
    output_path = pathlib.Path(output_dir) / f'w={window_size}_n={max_seqlets}_l={n_leiden}.h5'
    modiscolite.io.save_hdf5(output_path, pos_patterns, neg_patterns, window_size)
