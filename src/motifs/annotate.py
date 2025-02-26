import os
import os.path as osp
from typing import cast, Literal

import h5py
import numpy as np
from omegaconf import OmegaConf

from src.data import SeqDataset
from src.types import MotifInstance, t_seqlet_dict, ExpConfig, AllMotifInstances
from src.types import PatternConfig
from src.utils import load_config


def trim(
        cwm: np.ndarray,
        trim_threshold: float = 0.3,
) -> tuple[int, int]:
    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh_fwd = np.max(score) * trim_threshold
    pass_inds_fwd = np.where(score >= trim_thresh_fwd)[0]
    start, end = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score))
    return start, end


def annotate_patterns(
        exp_dir: str,
        split: Literal['train', 'val', 'test', 'all'],
        h5py_path: str,
        annot_yaml_path: str,
        output_dir: str,
):
    config = load_config(osp.join(exp_dir, 'config.yaml'), ExpConfig)
    dataset = SeqDataset(
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

    f = h5py.File(h5py_path, 'r')
    pattern_config = load_config(annot_yaml_path, PatternConfig)

    offset = (pattern_config.actual_window_size - pattern_config.trim_window_size) // 2
    patterns = pattern_config.patterns

    # pattern_dict: pattern name -> seqlets instances
    pattern_dict: dict[str, t_seqlet_dict] = cast(dict[str, t_seqlet_dict], {})
    for pattern_spec in patterns:
        pattern_name = pattern_spec.name
        is_forward = pattern_spec.is_forward
        key1, key2 = pattern_spec.key.split('.')
        pattern = f[key1][key2]
        seqlets: t_seqlet_dict = pattern['seqlets']
        if pattern_name not in pattern_dict:
            pattern_dict[pattern_name] = {}
        for k, v in seqlets.items():
            k: Literal[
                'n_seqlets', 'example_idx', 'sequence', 'contrib_scores', 'hypothetical_contribs', 'start', 'end', 'is_revcomp']
            v = v[()]
            if k == 'start' or k == 'end':
                v = v + offset
            elif k == 'is_revcomp':
                if not is_forward:
                    v = ~v

            if k not in pattern_dict[pattern_name]:
                pattern_dict[pattern_name][k] = v
            else:
                if k == 'n_seqlets':
                    pattern_dict[pattern_name][k] += v
                else:
                    pattern_dict[pattern_name][k] = np.concatenate([pattern_dict[pattern_name][k], v])

    # Trim seqlets
    for pattern_name in pattern_dict.keys():
        for i, cwm in enumerate(pattern_dict[pattern_name]['contrib_scores']):
            cwm: np.ndarray
            start_rel, end_rel = trim(cwm)
            start_original = pattern_dict[pattern_name]['start'][i]
            start_abs = start_original + start_rel
            end_abs = start_original + end_rel
            assert start_original <= start_abs < end_abs <= start_original + len(cwm)

            pattern_dict[pattern_name]['start'][i] = start_abs
            pattern_dict[pattern_name]['end'][i] = end_abs

    all_motifs = []
    global_id = 0
    for pattern_name, seqlets in pattern_dict.items():
        for i in range(seqlets['n_seqlets'].item()):
            start = seqlets['start'][i].item()
            end = seqlets['end'][i].item()
            example_idx = seqlets['example_idx'][i].item()
            is_revcomp = seqlets['is_revcomp'][i].item()
            chrom = dataset[example_idx]['chr']

            all_motifs.append(MotifInstance(
                global_idx=global_id,
                seqlet_idx=i,
                example_idx=example_idx,
                motif_name=pattern_name,
                chr=chrom,
                start=start,
                end=end,
                is_rev_comp=is_revcomp
            ).to_dict())

            global_id += 1

    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f"{split}_motif_instances.yaml")
    print(f"Saving motif instances to {output_path}", flush=True)
    schema = OmegaConf.structured(AllMotifInstances)
    validated = OmegaConf.merge(schema, {'instances': all_motifs})
    OmegaConf.save(validated, output_path)
