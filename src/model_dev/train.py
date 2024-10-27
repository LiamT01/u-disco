import os
import os.path as osp
from contextlib import nullcontext
from datetime import datetime
from typing import Tuple

import torch
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from src.data import DisPDataset
from src.metrics import pearson_corr
from src.models import DNASeqModel
from src.types import ExpConfig, t_dataset_item
from src.utils import set_device, get_logger, get_num_digits


def run_single_epoch(
        train: bool,
        config: ExpConfig,
        epoch: int,
        dataloader: DataLoader,
        model: DNASeqModel,
        optimizer: AdamW | None,
        ema: ExponentialMovingAverage,
        device: torch.device,
) -> Tuple[float, float]:
    assert not train or optimizer is not None, "Optimizer must be provided for training"

    if train:
        model.train()
        print(f"[Epoch {epoch + 1}]", flush=True)
    else:
        model.eval()

    all_loss = 0.
    all_pearson = 0.
    with ema.average_parameters() if not train else nullcontext():
        with torch.inference_mode() if not train else nullcontext():
            for i, batch in enumerate(dataloader):
                batch: t_dataset_item
                profile = batch["profile"].to(device)
                features = batch["features"].to(device)
                control = batch["control"].to(device) if config.model_dev.use_control else None

                results = model.run_batch(
                    features,
                    profile,
                    control,
                    use_prior=config.model_dev.use_prior if train else False,
                    reg_loss_weight=config.model_dev.reg_loss_weight if train else 0,
                    return_probs=True,
                )
                loss: torch.Tensor = results['loss'] + results['reg_loss']

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ema.update()

                all_loss += loss.item()

                pearson = pearson_corr(
                    results['profile'].detach().cpu().numpy(),
                    profile.cpu().numpy()
                ).mean().item()
                all_pearson += pearson

                if train:
                    print(f"\t[iter {i + 1}/{len(dataloader)}] train loss: {loss.item():.8f}, PCC: {pearson:.8f}",
                          flush=True)
                    wandb.log({
                        "train_loss_batch": loss.item(),
                        "train_pearson_batch": pearson,
                        "epoch": epoch + 1,
                    })

    all_loss /= len(dataloader)
    all_pearson /= len(dataloader)

    return all_loss, all_pearson


def train_epochs(config: ExpConfig) -> str:
    wandb.login()
    wandb.init(
        name=f"{config.model_dev.backend_model_class}-{config.raw_data.context_len / 1000}k-"
             f"{config.raw_data.cell_type}-split_{config.model_dev.seed}",
        project="disp",
    )

    train_set = DisPDataset(
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
        split='train',
        jitter_max=config.model_dev.jitter_max,
        reverse_complement_p=config.model_dev.reverse_complement_p,
    )

    val_set = DisPDataset(
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
        split='val',
        jitter_max=0,
        reverse_complement_p=0,
    )

    train_loader = DataLoader(train_set, batch_size=config.model_dev.bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.model_dev.bs, shuffle=False)

    model = DNASeqModel.from_config(config)

    optimizer = AdamW(model.parameters(), lr=config.model_dev.lr)

    device = set_device(config.model_dev.seed)

    output_dir = f'exp/train_{datetime.now():%Y-%m-%d_%H:%M:%S}'
    ckpt_dir = osp.join(output_dir, 'checkpoints')
    logger = get_logger(osp.join(output_dir, 'train.log'))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the config
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config, f)

    logger.info(model)

    best_val_loss = float('inf')

    model = model.float().to(device)

    # ema must follow to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99, use_num_updates=True)

    num_digits = get_num_digits(config.model_dev.n_epochs)
    for epoch in tqdm(range(config.model_dev.n_epochs)):
        train_loss, train_pearson = run_single_epoch(
            train=True,
            config=config,
            epoch=epoch,
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            ema=ema,
            device=device,
        )

        val_loss, val_pearson = run_single_epoch(
            train=False,
            config=config,
            epoch=epoch,
            dataloader=val_loader,
            model=model,
            optimizer=None,
            ema=ema,
            device=device,
        )

        logger.info(
            f"[Epoch {epoch + 1}] "
            f"train loss: {train_loss:.8f}, "
            f"train pearson: {train_pearson:.8f}, "
            f"val loss: {val_loss:.8f}, "
            f"val pearson: {val_pearson:.8f}"
        )
        wandb.log({
            "train_loss_epoch": train_loss,
            "train_pearson_epoch": train_pearson,
            "val_loss_epoch": val_loss,
            "val_pearson_epoch": val_pearson,
            "epoch": epoch + 1
        })

        if val_loss < best_val_loss:
            logger.info(f'\t\tBest val_loss={val_loss} so far was found! Model weights were saved.')
            with ema.average_parameters():
                torch.save(
                    model.state_dict(),
                    osp.join(ckpt_dir, f'epoch_{epoch:0{num_digits}d}.pth')
                )

            best_val_loss = val_loss

    return output_dir
