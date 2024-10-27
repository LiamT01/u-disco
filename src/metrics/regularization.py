# Adapted from: https://github.com/amtseng/fourier_attribution_priors/blob/master/src/model/profile_models.py
import numpy as np
import scipy
import torch


def fourier_att_prior_loss(
        status: torch.Tensor,
        input_grads: torch.Tensor,
        freq_limit: int,
        limit_softness: float,
        att_prior_grad_smooth_sigma: float,
):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    :param status: a B-tensor, where B is the batch size; each entry is 1 if
            that example is to be treated as a positive example, and 0 otherwise
    :param input_grads: a B x L x 4 tensor, where B is the batch size, L is
            the length of the input; this needs to be the gradients of the
            input with respect to the output; this should be
            *gradient times input*
    :param freq_limit: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
    :param limit_softness: amount to soften the limit by, using a hill
            function; None means no softness
    :param att_prior_grad_smooth_sigma: amount to smooth the gradient before
            computing the loss
    :return: a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = torch.sum(torch.abs(input_grads), dim=2)

    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(
        abs_grads, att_prior_grad_smooth_sigma
    )

    # Only do the positives
    pos_grads = grads_smooth[status == 1]

    # Loss for positives
    if pos_grads.nelement():
        pos_fft = torch.view_as_real(torch.fft.rfft(pos_grads, dim=-1))

        pos_mags = torch.norm(pos_fft, dim=2)
        pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
        pos_mags = pos_mags / pos_mag_sum

        # Cut off DC
        pos_mags = pos_mags[:, 1:]

        # Construct weight vector
        weights = torch.ones_like(pos_mags, device=pos_mags.device)
        if limit_softness is None:
            weights[:, freq_limit:] = 0
        else:
            x = torch.arange(
                1,
                pos_mags.size(1) - freq_limit + 1,
                device=pos_mags.device,
            ).float()
            weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights

        # Add up along frequency axis to get score
        pos_score = torch.sum(pos_weighted_mags, dim=1)
        pos_loss = 1 - pos_score
        return torch.mean(pos_loss)
    else:
        return torch.zeros(1, device=pos_grads.device)


def smooth_tensor_1d(
        input_tensor: torch.Tensor,
        smooth_sigma: float,
) -> torch.Tensor:
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e., the smoothing window is 1 + (2 * sigma));
            sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = torch.tensor(kernel, device=input_tensor.device)

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(
        input_tensor, kernel, padding=sigma
    )

    return torch.squeeze(smoothed, dim=1)
