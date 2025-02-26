# Adapted from: https://github.com/amtseng/fourier_attribution_priors/blob/master/src/model/profile_performance.py
import numpy as np
import scipy
import torch


def multinomial_log_probs(
        category_log_probs: torch.Tensor,
        trials: torch.Tensor,
        query_counts: torch.Tensor,
) -> torch.Tensor:
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    :param category_log_probs: a D x N tensor containing log probabilities (base
            e) of seeing each of the N classes/categories
    :param trials: a D-tensor containing the total number of trials for each
            distribution (can be different numbers)
    :param query_counts: a D x N tensor containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
    :return: a D-tensor containing the log probabilities (base e) of each
    observed query with its corresponding distribution. Note that D can be
    replaced with any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)

    trials, query_counts = trials.float(), query_counts.float()
    log_n_fact = torch.lgamma(trials + 1)
    log_counts_fact = torch.lgamma(query_counts + 1)
    log_counts_fact_sum = torch.sum(log_counts_fact, dim=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise sum
    log_prob_pows_sum = torch.sum(log_prob_pows, dim=-1)

    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum


def profile_logits_to_log_probs(
        logit_pred_profs: torch.Tensor,
        axis: int = -1,
) -> torch.Tensor:
    """
    Converts the model's predicted profile logits into normalized probabilities
    via a softmax on the specified dimension (defaults to axis=2).
    :param logit_pred_profs: a tensor/array containing the predicted profile logits
    :param axis: the dimension to apply the log-softmax on
    :return: a tensor/array of the same shape, containing the predicted profiles
    as log probabilities by doing a log softmax on the specified dimension. If
    the input is a tensor, the output will be a tensor. If the input is a NumPy
    array, the output will be a NumPy array. Note that the  reason why this
    function returns log probabilities rather than raw probabilities is for
    numerical stability.
    """
    if type(logit_pred_profs) is np.ndarray:
        return logit_pred_profs - \
            scipy.special.logsumexp(logit_pred_profs, axis=axis, keepdims=True)
    else:
        return torch.log_softmax(logit_pred_profs, dim=axis)


def pearson_corr(
        arr1: np.ndarray,
        arr2: np.ndarray,
) -> np.ndarray:
    """
    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.
    """
    mean1 = np.mean(arr1, axis=-1, keepdims=True)
    mean2 = np.mean(arr2, axis=-1, keepdims=True)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)

    # # Divide numerator by denominator, but use NaN where the denominator is 0
    # return np.divide(
    #     numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
    # )

    # Divide numerator by denominator, but use 1 where the denominator is 0
    return np.divide(
        numer, denom, out=np.full_like(numer, 1), where=(denom != 0)
    )
