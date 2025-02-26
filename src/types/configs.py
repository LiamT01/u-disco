from dataclasses import dataclass
from typing import List, TypeVar

T = TypeVar('T')


@dataclass
class RawDataConfig:
    cell_type: str
    split_dir: str
    context_len: int
    held_out_chrs: List[str]
    train_ratio: float
    bed_path: str
    bed_columns: List[str]
    profile_paths: List[str]
    control_paths: List[str]
    atac_paths: List[str]
    genome_path: str
    chr_refseq: dict[str, str]
    chr_lengths: dict[str, int]


@dataclass
class ModelDevConfig:
    seed: int

    use_atac: bool
    use_control: bool

    backend_model_module: str
    backend_model_class: str
    linear_upsample: bool | None
    dropout_p: float | None
    vocab_size: int
    n_epi: int
    noise_std: float

    n_epochs: int
    lr: float
    bs: int

    jitter_max: int
    reverse_complement_p: float

    use_prior: bool
    reg_loss_weight: float


@dataclass
class ExpConfig:
    raw_data: RawDataConfig
    model_dev: ModelDevConfig
