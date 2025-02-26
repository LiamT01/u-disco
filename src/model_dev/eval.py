import os.path as osp
from glob import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DisPDataset
from src.metrics import pearson_corr
from src.models import DNASeqModel
from src.types import ExpConfig, t_dataset_item
from src.utils import set_device


def eval_checkpoint(
        config: ExpConfig,
        test_loader: DataLoader,
        model: DNASeqModel,
        checkpoint_path: str,
        device: torch.device,
) -> tuple[float, float, float]:
    weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    model.eval()
    test_loss = 0.
    test_pearson = 0.
    test_pearson_list = []
    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            batch: t_dataset_item
            profile = batch["profile"].to(device)
            features = batch["features"].to(device)
            control = batch["control"].to(device) if config.model_dev.use_control else None

            results = model.run_batch(
                features,
                profile,
                control,
                use_prior=False,
                reg_loss_weight=0,
                return_probs=True,
            )

            loss: torch.Tensor = results['loss'] + results['reg_loss']
            test_loss += loss.item()

            pearson = pearson_corr(
                results['profile'].detach().cpu().numpy(),
                profile.cpu().numpy()
            )
            test_pearson += pearson.mean().item()
            test_pearson_list.append(pearson)

    test_loss /= len(test_loader)
    test_pearson /= len(test_loader)
    test_pearson_std = np.std(np.concatenate(test_pearson_list)).item()

    return test_loss, test_pearson, test_pearson_std


def eval_exp(
        exp_dir: str,
        config: ExpConfig,
        overwrite_atac_paths: list[str] | None = None,
):
    atac_path_list: List[str] | None = None
    if config.model_dev.use_atac:
        atac_path_list = overwrite_atac_paths if overwrite_atac_paths is not None \
            else config.raw_data.atac_paths

    test_set = DisPDataset(
        seed=config.model_dev.seed,
        split_dir=config.raw_data.split_dir,
        context_len=config.raw_data.context_len,
        held_out_chrs=config.raw_data.held_out_chrs,
        train_ratio=config.raw_data.train_ratio,
        bed_path=config.raw_data.bed_path,
        bed_columns=config.raw_data.bed_columns,
        profile_paths=config.raw_data.profile_paths,
        control_paths=config.raw_data.control_paths,
        atac_paths=atac_path_list,
        genome_path=config.raw_data.genome_path,
        chr_refseq=config.raw_data.chr_refseq,
        chr_lengths=config.raw_data.chr_lengths,
        split='test',
        jitter_max=0,
        reverse_complement_p=0,
    )
    test_loader = DataLoader(test_set, batch_size=config.model_dev.bs, shuffle=False)

    device = set_device(config.model_dev.seed)
    model = DNASeqModel.from_config(config).to(device).float()

    output_path = osp.join(exp_dir, "eval.log")
    print(f"Writing evaluation results to {output_path}", flush=True)
    for checkpoint_path in tqdm(sorted(glob(osp.join(exp_dir, "checkpoints", "*.pth")))):
        loss, pcc, pcc_std = eval_checkpoint(config, test_loader, model, checkpoint_path, device)
        with open(output_path, "a") as f:
            f.write(f"{checkpoint_path}\n")
            f.write(f"\tTest loss: {loss:.8f}\n"
                    f"\tPCC: {pcc:.8f} (std: {pcc_std:.8f})\n")
