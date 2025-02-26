from typing import Literal, List, cast

import numpy as np
import pandas as pd
import pyBigWig
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import Dataset

from src.types import t_dataset_item, t_dataset_item_wo_control
from src.utils import calc_avg_signal, get_split_indices, tokenizer, register_cache


class SeqDataset(Dataset):
    context_len: int
    bed: pd.DataFrame
    peak_ids: List[int]
    profile_bws: List[pyBigWig]
    control_bws: List[pyBigWig] | None
    atac_bws: List[pyBigWig] | None
    jitter_max: int
    reverse_complement_p: float
    chr_to_fasta: dict[str, SeqIO.SeqRecord]
    chr_lengths: dict[str, int]

    def __init__(
            self,
            seed: int,
            split_dir: str,
            context_len: int,
            held_out_chrs: List[str],
            train_ratio: float,
            bed_path: str,
            bed_columns: List[str],
            profile_paths: List[str],
            control_paths: List[str] | None,
            atac_paths: List[str] | None,
            genome_path: str,
            chr_refseq: dict[str, str],
            chr_lengths: dict[str, int],
            split: Literal['train', 'val', 'test', 'all'],
            jitter_max: int,
            reverse_complement_p: float,
    ):
        super().__init__()
        assert control_paths is None or len(
            control_paths) > 0, "control_path_list, when not None, must be a non-empty list"
        assert atac_paths is None or len(
            atac_paths) > 0, "atac_path_list, when not None, must be a non-empty list"
        assert split in ["train", "val", "test", "all"], "split must be one of 'train', 'val', 'test', or 'all'"

        self.context_len = context_len

        self.bed = register_cache(
            key=bed_path,
            callback=lambda: pd.read_csv(bed_path, delimiter="\t", header=None),
        )
        assert self.bed.shape[1] == len(bed_columns), f"Bed file must have {len(bed_columns)} columns"
        self.bed.columns = bed_columns

        split_ids = get_split_indices(
            split_dir=split_dir,
            bed=self.bed,
            held_out_chrs=held_out_chrs,
            train_ratio=train_ratio,
            context_len=context_len,
            chr_lengths=chr_lengths,
            seed=seed,
        )

        if split == 'all':
            self.peak_ids = sorted(split_ids["train"] + split_ids["val"] + split_ids["test"])
        else:
            self.peak_ids = sorted(split_ids[cast(Literal['train', 'val', 'test'], split)])

        self.profile_bws = register_cache(
            key=' '.join(profile_paths),
            callback=lambda: [pyBigWig.open(path) for path in profile_paths],
        )
        self.control_bws = register_cache(
            key=' '.join(control_paths),
            callback=lambda: [pyBigWig.open(path) for path in control_paths],
        ) if control_paths is not None else None
        self.atac_bws = register_cache(
            key=' '.join(atac_paths),
            callback=lambda: [pyBigWig.open(atac_path) for atac_path in atac_paths],
        ) if atac_paths is not None else None

        self.jitter_max = jitter_max
        self.reverse_complement_p = reverse_complement_p

        def preprocess_genome():
            record_dict = SeqIO.index(genome_path, "fasta")
            return {k: record_dict[v] for k, v in chr_refseq.items()}

        self.chr_to_fasta = register_cache(
            key=genome_path,
            callback=preprocess_genome,
        )

        self.chr_lengths = chr_lengths

    def __len__(self):
        return len(self.peak_ids)

    def __getitem__(self, idx: int) -> t_dataset_item | t_dataset_item_wo_control:
        peak_id = self.peak_ids[idx]
        peak = self.bed.iloc[peak_id]

        peak_start = int(cast(np.int64, peak['start']))
        peak_end = int(cast(np.int64, peak['end']))
        chr_name = str(peak['chr'])

        chr_seq = self.chr_to_fasta[chr_name]

        mid = (peak_start + peak_end) // 2
        start = max(mid - self.context_len // 2, 0)
        end = min(mid + self.context_len // 2, len(chr_seq))

        assert end - start == self.context_len

        if self.jitter_max > 0:
            jitter = np.random.randint(-self.jitter_max, self.jitter_max)

            max_space = (-start, self.chr_lengths[chr_name] - end)
            if jitter < max_space[0]:
                jitter = max_space[0]
            elif jitter > max_space[1]:
                jitter = max_space[1]

            start = start + jitter
            end = end + jitter
            assert end - start == self.context_len

        signal = calc_avg_signal(chr_name, start, end, self.profile_bws)

        control = None
        if self.control_bws is not None:
            control = calc_avg_signal(chr_name, start, end, self.control_bws)

        seq = str(chr_seq[start:end].seq).upper()

        atac = None
        if self.atac_bws is not None:
            atac = calc_avg_signal(chr_name, start, end, self.atac_bws).reshape(-1, 1)

        if self.reverse_complement_p > 0 and np.random.rand() < self.reverse_complement_p:
            seq = seq[::-1].translate(str.maketrans("ACGT", "TGCA"))
            if self.atac_bws is not None:
                atac = atac[::-1].copy()
            signal = signal[::-1].copy()

        one_hot = F.one_hot(torch.LongTensor([tokenizer[x] for x in seq]), num_classes=5).float()
        features = one_hot if self.atac_bws is None else torch.cat([one_hot, torch.FloatTensor(atac)], dim=-1)

        if control is not None:
            return {
                "features": torch.FloatTensor(features),
                "profile": torch.FloatTensor(signal),
                "control": torch.FloatTensor(control),
                "peak_id": peak_id,
                "peak_start": peak_start,
                "peak_end": peak_end,
                "start": start,
                "end": end,
                "chr": chr_name,
            }
        else:
            return {
                "features": torch.FloatTensor(features),
                "profile": torch.FloatTensor(signal),
                "peak_id": peak_id,
                "peak_start": peak_start,
                "peak_end": peak_end,
                "start": start,
                "end": end,
                "chr": chr_name,
            }
