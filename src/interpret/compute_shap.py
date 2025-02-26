import os
import os.path as osp
from typing import Literal, cast, List

import numpy as np
import shap
import torch
from tqdm import tqdm

from src.data import DisPDataset
from src.models import DNASeqModel
from src.types import ExpConfig, t_dataset_item
from src.utils import set_device, load_config
from .background import create_input_seq_background, create_profile_control_background
from .model_wrapper import WrapperProfileModel


def interpret_model(
        split: Literal["train", "val", "test", "all"],
        exp_dir: str,
        ckpt_name: str,
        seed: int,
        n_background: int = 100,
        n_samples: int = 200,
        batch_size: int = 10,
):
    device = set_device(seed)
    ckpt_path = osp.join(exp_dir, "checkpoints", ckpt_name)

    output_dir = osp.join(exp_dir, f"interpret-{split}")
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(osp.join(exp_dir, 'config.yaml'), ExpConfig)

    dataset = DisPDataset(
        seed=config.model_dev.seed,
        split_dir=config.raw_data.split_dir,
        context_len=config.raw_data.context_len,
        held_out_chrs=config.raw_data.held_out_chrs,
        train_ratio=config.raw_data.train_ratio,
        bed_path=config.raw_data.bed_path,
        bed_columns=config.raw_data.bed_columns,
        profile_paths=config.raw_data.profile_paths,
        control_paths=config.raw_data.control_paths,
        atac_paths=config.raw_data.atac_paths if config.model_dev.use_atac else None,
        genome_path=config.raw_data.genome_path,
        chr_refseq=config.raw_data.chr_refseq,
        chr_lengths=config.raw_data.chr_lengths,
        split=split,
        jitter_max=0,
        reverse_complement_p=0,
    )

    model = DNASeqModel.from_config(config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    wrapped_model = WrapperProfileModel(model)
    wrapped_model = wrapped_model.float().to(device).eval()

    ohe_seq_list: List[np.ndarray] = cast(List[np.ndarray], [])
    seq_shap_list: List[np.ndarray] = cast(List[np.ndarray], [])
    seq_shap_centered_list: List[np.ndarray] = cast(List[np.ndarray], [])
    for i in tqdm(range(len(dataset))):
        example: t_dataset_item = dataset[i]
        one_hot = example['features'][:, :4]
        control = example['control'] if config.model_dev.use_control else None

        seq_len, n_all_features = example['features'].shape

        seq_bg = create_input_seq_background(
            input_seq=one_hot,
            input_length=seq_len,
            device=device,
            bg_size=n_background,
            seed=i,
        )
        # Append fifth base (N for unknown) to the one-hot encoding
        # and any potential epigenetic features
        seq_bg = torch.cat([
            seq_bg,
            torch.zeros(seq_bg.shape[0], seq_bg.shape[1], n_all_features - seq_bg.shape[2], device=seq_bg.device),
        ], dim=-1)
        # Shape: G x I x (4 + 1 + ANY EPIGENETIC FEATURES)

        control_bg = create_profile_control_background(
            control_profs=control,
            profile_length=seq_len,
            device=device,
            bg_size=n_background,
        ) if config.model_dev.use_control else None

        e = shap.GradientExplainer(wrapped_model, data=[seq_bg, control_bg], batch_size=batch_size)

        features = example['features'].unsqueeze(0).to(device)
        # Shape: 1 x I x (4 + 1 + ANY EPIGENETIC FEATURES)

        control = control.unsqueeze(0).to(device) if config.model_dev.use_control else None
        # Shape: 1 x I

        shap_values = e.shap_values([features, control], nsamples=n_samples)
        # If list, extract element 0
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).squeeze()
        assert shap_values.shape == (seq_len, n_all_features)

        seq_shap = np.transpose(shap_values[:, :4]).astype(np.float32)
        assert seq_shap.shape == (4, seq_len)
        seq_shap_list.append(seq_shap)

        seq_shap_mean = seq_shap.mean(axis=0, keepdims=True)
        assert seq_shap_mean.shape == (1, seq_len)
        seq_shap_centered = seq_shap - seq_shap_mean
        assert seq_shap_centered.shape == (4, seq_len)
        seq_shap_centered_list.append(seq_shap_centered)

        ohe_seq = torch.transpose(one_hot, 0, 1).numpy().astype(np.float32)
        assert ohe_seq.shape == (4, seq_len)
        ohe_seq_list.append(ohe_seq)

    print(f"Saving one-hot encoded sequences and SHAP values to {output_dir}", flush=True)
    np.savez_compressed(osp.join(output_dir, "seq_shap.npz"), seq_shap_list)
    np.savez_compressed(osp.join(output_dir, "seq_shap_centered.npz"), seq_shap_centered_list)
    np.savez_compressed(osp.join(output_dir, "ohe_seq.npz"), ohe_seq_list)
