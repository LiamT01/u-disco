from typing import Tuple

import torch

from src.models import BPNet, UDisCo, DNASeqModel


def make_inputs(
        batch_size: int,
        in_channels: int,
        seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.randn((batch_size, in_channels, seq_len))
    profile = torch.randn((batch_size, seq_len))
    control = torch.randn((batch_size, seq_len))
    return features, profile, control


def test_bpnet():
    model = BPNet(
        in_channels=5,
        use_control=True,
    )
    features, profile, control = make_inputs(4, 5, 20000)
    out = model(features, control)
    assert out.shape == (4, 20000)


def test_udisco():
    model = UDisCo(
        in_channels=5,
        linear=True,
        dropout_p=0.1,
        use_control=True,
    )
    features, profile, control = make_inputs(4, 5, 20000)
    out = model(features, control)
    assert out.shape == (4, 20000)


def test_wrapper():
    model = DNASeqModel(
        vocab_size=5,
        n_epi=3,
        backend=UDisCo(in_channels=8, linear=True, dropout_p=0.1, use_control=True),
        noise_std=0.1,
    )
    features, profile, control = make_inputs(4, 8, 20000)
    out = model.run_batch(features, profile, control)
    assert out['profile'].shape == (4, 20000)
