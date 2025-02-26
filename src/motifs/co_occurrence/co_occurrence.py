import json
import os
import os.path as osp
from typing import List, Tuple, cast
from typing import TypedDict

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.types import t_grouped_motifs, MotifInstance, PatternConfig, CoOccurrenceConfig, Range, AllMotifInstances
from src.utils import map_tasks, load_config, get_num_workers, get_timestamp
from .query import query_tree_range
from .shuffle import get_shuffled_motifs_list


def group_motifs_by_name(motifs: List[MotifInstance]) -> t_grouped_motifs:
    results: t_grouped_motifs = {}
    for motif in motifs:
        motif_name = motif.motif_name
        chrom = motif.chr
        if motif_name not in results:
            results[motif_name] = {}
        if chrom not in results[motif_name]:
            results[motif_name][chrom] = []
        results[motif_name][chrom].append(motif)
    return results


def count_close_not_close_kdtree(
        motifs_a: dict[str, List[MotifInstance]],
        motifs_b: dict[str, List[MotifInstance]],
        d_min: int,
        d_max: int,
        across_chr: bool,
) -> Tuple[int, int]:
    close_count: int = 0
    not_close_count: int = 0
    for chrom in motifs_a:
        if chrom not in motifs_b:
            continue

        positions_a = [(motif.start + motif.end) // 2 for motif in motifs_a[chrom]]
        positions_b = [(motif.start + motif.end) // 2 for motif in motifs_b[chrom]]

        indices_within = query_tree_range(positions_a, positions_b, d_min, d_max)
        close_count_chr = sum([len(_neighbors) for _neighbors in indices_within])
        close_count += close_count_chr
        if not across_chr:
            not_close_count_chr = len(positions_a) * len(positions_b) - close_count_chr
            not_close_count += not_close_count_chr

    if across_chr:
        not_close_count = len(motifs_a) * len(motifs_b) - close_count

    return close_count, not_close_count


t_count_for_one_arg = TypedDict('t_count_for_one_arg', {
    'motif_name_a': str,
    'motif_name_b': str,
    'motifs_a': dict[str, List[MotifInstance]],
    'motifs_b': dict[str, List[MotifInstance]],
    'd_min': int,
    'd_max': int,
})


def co_occurrence_for_one_motif_list(arg: t_count_for_one_arg) -> Tuple[int, int]:
    return count_close_not_close_kdtree(
        arg['motifs_a'], arg['motifs_b'],
        arg['d_min'], arg['d_max'],
        across_chr=False
    )


t_co_occur_res_pair = TypedDict('t_co_occur_res_pair', {
    'motif_pair': Tuple[str, str],
    'cont_matrix': List[List[float]],
})

t_co_occur_res_range = TypedDict('t_co_occur_res_range', {
    'd_range': Tuple[int, int],
    'res': List[t_co_occur_res_pair],
})

t_co_occur_res = List[t_co_occur_res_range]


def create_contingency_matrix(
        all_motifs: t_grouped_motifs,
        shuffled_motifs_list: List[t_grouped_motifs],
        motif_name_a: str,
        motif_name_b: str,
        d_min: int,
        d_max: int,
        num_trials: int = 100,
) -> List[List[float]]:
    assert len(shuffled_motifs_list) == num_trials

    # Calculate counts for original motifs
    close_count, not_close_count = count_close_not_close_kdtree(all_motifs[motif_name_a],
                                                                all_motifs[motif_name_b],
                                                                d_min, d_max, across_chr=False)

    res_shuffled = map_tasks(co_occurrence_for_one_motif_list, [
        {
            'motif_name_a': motif_name_a,
            'motif_name_b': motif_name_b,
            'motifs_a': motif_list[motif_name_a],
            'motifs_b': motif_list[motif_name_b],
            'd_min': d_min,
            'd_max': d_max,
        } for motif_list in shuffled_motifs_list
    ])

    res_shuffled_arr = np.array(res_shuffled)
    total_close_shuffled: int = res_shuffled_arr[:, 0].sum().item()
    total_not_close_shuffled: int = res_shuffled_arr[:, 1].sum().item()

    # Calculate average shuffled counts
    avg_close_shuffled = total_close_shuffled / num_trials
    avg_not_close_shuffled = total_not_close_shuffled / num_trials

    # Create contingency matrix
    cont_matrix = [
        [avg_not_close_shuffled, not_close_count],
        [avg_close_shuffled, close_count]
    ]

    return cont_matrix


def process_distance_range(
        d_min: int,
        d_max: int,
        motif_names: List[str],
        all_motifs: t_grouped_motifs,
        num_trials: int,
        shuffled_motifs_list: List[t_grouped_motifs],
) -> t_co_occur_res_range:
    # contingency_matrices: dict[Tuple[str, str], List[List[float]]] = {}
    contingency_matrices: t_co_occur_res_range = {
        'd_range': (d_min, d_max),
        'res': [],
    }

    total = len(motif_names) * (len(motif_names) - 1) // 2 + len(motif_names)
    with tqdm(total=total) as pbar:
        for motif_name_a in motif_names:
            for motif_name_b in motif_names:
                if (motif_name_a, motif_name_b) in contingency_matrices or (
                        motif_name_b, motif_name_a) in contingency_matrices:
                    continue

                pbar.set_description(f"[{d_min} - {d_max}] {motif_name_a} - {motif_name_b}")

                cont_matrix = create_contingency_matrix(all_motifs, shuffled_motifs_list,
                                                        motif_name_a, motif_name_b,
                                                        d_min=d_min, d_max=d_max,
                                                        num_trials=num_trials)
                contingency_matrices['res'].append({
                    'motif_pair': (motif_name_a, motif_name_b),
                    'cont_matrix': cont_matrix,
                })

                pbar.update(1)

    return contingency_matrices


def _co_occurrence(
        motif_names: List[str],
        all_motifs: t_grouped_motifs,
        shuffled_motifs_list: List[t_grouped_motifs],
        num_trials: int,
        distance_ranges: List[Range],
) -> t_co_occur_res:
    assert len(shuffled_motifs_list) == num_trials, "Number of shuffled motifs must match the number of trials"

    results: t_co_occur_res = cast(t_co_occur_res, [])

    for d_range in tqdm(distance_ranges):
        d_min, d_max = d_range.low, d_range.high
        contingency_matrices = process_distance_range(d_min, d_max, motif_names, all_motifs,
                                                      num_trials, shuffled_motifs_list)
        results.append(contingency_matrices)

    return results


def calc_co_occurrence(
        config_path: str,
        annot_yaml_path: str,
        motif_instances_yaml_path: str,
        output_dir: str,
):
    config = load_config(config_path, CoOccurrenceConfig)

    pattern_config = load_config(annot_yaml_path, PatternConfig)
    motif_names = list(set(x.name for x in pattern_config.patterns))

    all_motifs = load_config(motif_instances_yaml_path, AllMotifInstances).instances

    logger.info(f'Using {get_num_workers()} workers')

    logger.info('Shuffling motifs...')
    shuffled_motifs_list = get_shuffled_motifs_list(all_motifs, config.num_trials, seed=config.seed)

    logger.info('Grouping motifs...')
    all_grouped_motifs = group_motifs_by_name(all_motifs)
    grouped_shuffled_motifs_list = [group_motifs_by_name(motifs) for motifs in tqdm(shuffled_motifs_list)]

    logger.info('Calculating co-occurrence...')
    results = _co_occurrence(
        motif_names=motif_names,
        all_motifs=all_grouped_motifs,
        shuffled_motifs_list=grouped_shuffled_motifs_list,
        num_trials=config.num_trials,
        distance_ranges=config.d_ranges,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f'co_occurrence_results_{get_timestamp()}.json')
    print(f"Saving results to {output_path}", flush=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
