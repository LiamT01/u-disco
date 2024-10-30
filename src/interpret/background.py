# Adapted from: https://github.com/amtseng/fourier_attribution_priors/blob/master/src/extract/dinuc_shuffle.py

import numpy as np
import torch


def one_hot_to_tokens(one_hot: np.ndarray) -> np.ndarray:
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(
        tokens: np.ndarray,
        one_hot_dim: int,
) -> np.ndarray:
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(
        seq: np.ndarray,
        num_shuffles: int,
        rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    :param seq: L x D NumPy array of one-hot encodings
    :param num_shuffles: the number of shuffles to create, N
    :param rng: a NumPy RandomState object, to use for performing shuffles
    :return: N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded.
    """
    assert type(seq) is np.ndarray and len(seq.shape) == 2, "Expected one-hot encoded array"

    seq_len, one_hot_dim = seq.shape
    arr = one_hot_to_tokens(seq)

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    all_results = np.empty(
        (num_shuffles, seq_len, one_hot_dim), dtype=seq.dtype
    )

    for i in range(num_shuffles):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results


def create_input_seq_background(
        input_seq: torch.Tensor,
        input_length: int,
        device: torch.device,
        bg_size: int = 10,
        seed: int = 20200219,
) -> torch.Tensor:
    """
    From the input sequence to a model, generates a set of background
    sequences to perform interpretation against.
    :param input_seq: I x 4 tensor of one-hot encoded input sequence, or None
    :param input_length: length of input, I
    :param device:
    :param bg_size: the number of background examples to generate, G
    :param seed:
    :return: G x I x 4 tensor containing randomly dinucleotide-shuffles of the
    original input sequence. If `input_seq` is None, then a G x I x 4 tensor of
    all 0s is returned.
    """
    if input_seq is None:
        input_seq_bg_shape = (bg_size, input_length, 4)
        return torch.zeros(input_seq_bg_shape, device=device).float()

    # Do dinucleotide shuffles
    input_seq_np = input_seq.cpu().numpy()
    rng = np.random.RandomState(seed)
    input_seq_bg_np = dinuc_shuffle(input_seq_np, bg_size, rng=rng)
    return torch.from_numpy(input_seq_bg_np).to(device).float()


def create_profile_control_background(
        control_profs: torch.Tensor,
        profile_length: int,
        device: torch.device,
        bg_size: int = 10,
) -> torch.Tensor:
    """
    Generates a background for a set of profile controls. In general, this is
    the given control profiles, copied a number of times (i.e. the background
    for controls should always be the same). Note this is only used for profile
    models.
    :param control_profs: (T or 1) x O x S tensor of control profiles, or None
    :param profile_length: length of profile, O
    :param device:
    :param bg_size: the number of background examples to generate, G
    :return: the tensor of `control_profs`, replicated G times. If `controls` is
    "matched", this becomes a G x T x O x S tensor; if `controls` is "shared",
    this is a G x 1 x O x S tensor. If `control_profs` is None, then a tensor of
    all 0s is returned, whose shape is determined by `controls`.
    """
    control_profs_bg_shape = (bg_size, profile_length)
    if control_profs is None:
        return torch.zeros(control_profs_bg_shape, device=device).float()

    # Replicate `control_profs`
    return torch.stack([control_profs.squeeze()] * bg_size, dim=0).to(device).float()
