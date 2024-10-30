import random
from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm

from src.types import MotifInstance


def shuffle_motifs_by_chr(all_motifs: List[MotifInstance]) -> List[MotifInstance]:
    positions_by_chr: defaultdict[str, List[int]] = defaultdict(list)
    motifs_by_chr: defaultdict[str, List[MotifInstance]] = defaultdict(list)

    # Collect positions and motifs by chromosome
    for motif in all_motifs:
        mid_pos = (motif.start + motif.end) // 2
        positions_by_chr[motif.chr].append(mid_pos)
        motifs_by_chr[motif.chr].append(motif)

    shuffled_motifs: List[MotifInstance] = []
    for chrom in positions_by_chr:
        random.shuffle(motifs_by_chr[chrom])
        for i, motif in enumerate(motifs_by_chr[chrom]):
            shuffled_motif = motif.copy()
            shuffled_motif.start = positions_by_chr[chrom][i] - (motif.end - motif.start) // 2
            shuffled_motif.end = positions_by_chr[chrom][i] + (motif.end - motif.start) // 2
            shuffled_motifs.append(shuffled_motif)

    return shuffled_motifs


def get_shuffled_motifs_list(
        all_motifs: List[MotifInstance],
        num_trials: int,
        seed: int = 0
) -> List[List[MotifInstance]]:
    random.seed(seed)
    np.random.seed(seed)

    shuffled_motifs_list = [shuffle_motifs_by_chr(all_motifs) for _ in tqdm(range(num_trials))]
    return shuffled_motifs_list
