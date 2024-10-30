from dataclasses import dataclass
from typing import List, TypedDict, Literal

import numpy as np


@dataclass
class Pattern:
    name: str
    key: str
    is_forward: bool


@dataclass
class PatternConfig:
    actual_window_size: int
    trim_window_size: int
    patterns: List[Pattern]


@dataclass
class MotifInstance:
    global_idx: int
    seqlet_idx: int
    example_idx: int
    motif_name: str
    chr: str
    start: int
    end: int
    is_rev_comp: bool

    def to_dict(self):
        return {
            "global_idx": self.global_idx,
            "seqlet_idx": self.seqlet_idx,
            "example_idx": self.example_idx,
            "motif_name": self.motif_name,
            "chr": self.chr,
            "start": self.start,
            "end": self.end,
            "is_rev_comp": self.is_rev_comp,
        }

    def copy(self):
        return MotifInstance(
            global_idx=self.global_idx,
            seqlet_idx=self.seqlet_idx,
            example_idx=self.example_idx,
            motif_name=self.motif_name,
            chr=self.chr,
            start=self.start,
            end=self.end,
            is_rev_comp=self.is_rev_comp,
        )


@dataclass
class AllMotifInstances:
    instances: List[MotifInstance]


t_seqlet_dict = TypedDict('t_seqlet_dict', {
    'n_seqlets': np.ndarray,
    'example_idx': np.ndarray,
    'sequence': np.ndarray,
    'contrib_scores': np.ndarray,
    'hypothetical_contribs': np.ndarray,
    'start': np.ndarray,
    'end': np.ndarray,
    'is_revcomp': np.ndarray,
})

t_motif_orient = Literal[0, 1]

# motif_name -> chrom -> motifs
t_grouped_motifs = dict[str, dict[str, List[MotifInstance]]]

# motif_name -> orient -> chrom -> motifs
t_grouped_orient_motifs = dict[str, dict[t_motif_orient, dict[str, List[MotifInstance]]]]


@dataclass
class Range:
    low: int
    high: int


@dataclass
class CoOccurrenceConfig:
    num_trials: int
    d_ranges: List[Range]
    seed: int
