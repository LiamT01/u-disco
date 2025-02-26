import json
import os
from typing import List, Tuple, TypedDict, cast, Literal

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.types import (
    MotifInstance,
    t_motif_orient,
    t_grouped_orient_motifs,
    CoOccurrenceConfig,
    PatternConfig,
    Range, AllMotifInstances,
)
from src.utils import map_tasks, load_config, get_num_workers, get_timestamp
from .query import query_tree_range
from .shuffle import get_shuffled_motifs_list


def group_motifs_by_name_orient(motifs: List[MotifInstance]) -> t_grouped_orient_motifs:
    results: t_grouped_orient_motifs = {}
    for motif in motifs:
        motif_name = motif.motif_name
        orient: t_motif_orient = 0 if motif.is_rev_comp else 1
        chrom = motif.chr
        if motif_name not in results:
            results[motif_name] = {}
        if orient not in results[motif_name]:
            results[motif_name][orient] = {}
        if chrom not in results[motif_name][orient]:
            results[motif_name][orient][chrom] = []
        results[motif_name][orient][chrom].append(motif)
    return results


def count_pairs_kdtree(
        motifs_a: dict[str, List[MotifInstance]],
        motifs_b: dict[str, List[MotifInstance]],
        d_min: int,
        d_max: int,
        a_smaller_pos: bool = False,
) -> Tuple[int, int]:
    close_count: int = 0
    not_close_count: int = 0
    for chrom in motifs_a:
        if chrom not in motifs_b:
            continue

        positions_a = [(motif.start + motif.end) // 2 for motif in motifs_a[chrom]]
        positions_b = [(motif.start + motif.end) // 2 for motif in motifs_b[chrom]]

        global_ids_a = [motif.global_idx for motif in motifs_a[chrom]]
        global_ids_b = [motif.global_idx for motif in motifs_b[chrom]]

        indices_within = query_tree_range(positions_a, positions_b, d_min, d_max)
        close_count_chr: int = 0
        for a, _neighbors in enumerate(indices_within):
            for b in _neighbors:
                if global_ids_a[a] == global_ids_b[b]:
                    continue
                if ((a_smaller_pos and positions_a[a] < positions_b[b]) or
                        (not a_smaller_pos and positions_a[a] > positions_b[b])):
                    close_count_chr += 1

        close_count += close_count_chr
        not_close_count_chr = len(positions_a) * len(positions_b) - close_count_chr
        not_close_count += not_close_count_chr

    return close_count, not_close_count


t_unit_res = TypedDict('t_unit_res', {
    'cont_matrix': List[List[float]],
    'shuffled_counts': List[List[float]],
})
t_orient_res = TypedDict('t_orient_res', {
    'orient': Tuple[int, int],
    'res': t_unit_res,
})
t_pair_orient_res = dict[Tuple[str, str], dict[Tuple[int, int], t_unit_res]]
t_pair_res = TypedDict('t_pair_res', {
    'motif_pair': Tuple[str, str],
    'res': List[t_orient_res],
})
# t_preference_res = dict[Tuple[int, int], t_pair_orient_res]
t_preference_res = List[TypedDict('t_preference_res', {
    'd_range': Tuple[int, int],
    'res': List[t_pair_res],
})]

t_orient_count_for_one_arg = TypedDict('t_orient_count_for_one_arg', {
    'motif_name_a': str,
    'motif_name_b': str,
    'motifs_a': dict[str, List[MotifInstance]],
    'motifs_b': dict[str, List[MotifInstance]],
    'orient_a': Literal[0, 1],
    'orient_b': Literal[0, 1],
    'd_min': int,
    'd_max': int,
})


def preference_for_one_motif_list(arg: t_orient_count_for_one_arg) -> Tuple[int, int]:
    return count_pairs_kdtree(
        arg['motifs_a'], arg['motifs_b'],
        arg['d_min'], arg['d_max'],
        a_smaller_pos=True
    )


def _preference(
        motif_names: List[str],
        all_motifs: t_grouped_orient_motifs,
        shuffled_motifs_list: List[t_grouped_orient_motifs],
        num_trials: int,
        distance_ranges: List[Range],
) -> t_preference_res:
    assert len(shuffled_motifs_list) == num_trials, "Number of shuffled motifs must match the number of trials"

    results: t_preference_res = cast(t_preference_res, [])

    for d_range in tqdm(distance_ranges):
        d_min, d_max = d_range.low, d_range.high
        pair_orient: t_pair_orient_res = {}

        total = len(motif_names) * len(motif_names)
        with tqdm(total=total) as pbar:
            for motif_name_a in motif_names:
                for motif_name_b in motif_names:
                    pbar.set_description(f"[{d_min} - {d_max}] {motif_name_a} - {motif_name_b}")

                    if (motif_name_a, motif_name_b) not in pair_orient:
                        pair_orient[(motif_name_a, motif_name_b)] = {}

                    for orient_a in [1, 0]:
                        for orient_b in [1, 0]:
                            close_count, not_close_count = count_pairs_kdtree(
                                all_motifs[motif_name_a][cast(t_motif_orient, orient_a)],
                                all_motifs[motif_name_b][cast(t_motif_orient, orient_b)],
                                d_min, d_max,
                                a_smaller_pos=True,
                            )

                            res_shuffled = map_tasks(preference_for_one_motif_list, [
                                {
                                    'motif_name_a': motif_name_a,
                                    'motif_name_b': motif_name_b,
                                    'motifs_a': motif_list[motif_name_a][cast(t_motif_orient, orient_a)],
                                    'motifs_b': motif_list[motif_name_b][cast(t_motif_orient, orient_b)],
                                    'orient_a': cast(t_motif_orient, orient_a),
                                    'orient_b': cast(t_motif_orient, orient_b),
                                    'd_min': d_min,
                                    'd_max': d_max,
                                } for motif_list in shuffled_motifs_list
                            ])
                            res_shuffled_arr: np.ndarray = np.array(res_shuffled)
                            total_close_shuffled: int = res_shuffled_arr[:, 0].sum().item()
                            total_not_close_shuffled: int = res_shuffled_arr[:, 1].sum().item()

                            avg_close_shuffled = total_close_shuffled / num_trials
                            avg_not_close_shuffled = total_not_close_shuffled / num_trials

                            pair_orient[(motif_name_a, motif_name_b)][(orient_a, orient_b)] = {
                                'cont_matrix': [
                                    [avg_not_close_shuffled, not_close_count],
                                    [avg_close_shuffled, close_count]
                                ],
                                'shuffled_counts': res_shuffled_arr.tolist(),
                            }

                    pbar.update(1)

        results.append({
            'd_range': (d_min, d_max),
            'res': [
                {
                    'motif_pair': pair,
                    'res': [
                        {
                            'orient': orient,
                            'res': res,
                        } for orient, res in pair_res.items()
                    ],
                } for pair, pair_res in pair_orient.items()
            ],
        })

    return results


def calc_preference(
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
    all_grouped_motifs = group_motifs_by_name_orient(all_motifs)
    grouped_shuffled_motifs_list = [group_motifs_by_name_orient(motifs) for motifs in tqdm(shuffled_motifs_list)]

    logger.info('Calculating preference...')
    results = _preference(
        motif_names=motif_names,
        all_motifs=all_grouped_motifs,
        shuffled_motifs_list=grouped_shuffled_motifs_list,
        num_trials=config.num_trials,
        distance_ranges=config.d_ranges,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"preference_results_{get_timestamp()}.json")
    print(f"Saving results to {output_path}", flush=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
