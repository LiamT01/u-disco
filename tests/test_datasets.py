from src.data import DisPDataset
from src.types import RawDataConfig
from src.utils import load_config


def test_disp():
    raw_data_config = load_config('configs/raw_data/disp_seq_sknmc.yaml', RawDataConfig)

    dataset = DisPDataset(
        seed=0,
        split_dir=raw_data_config.split_dir,
        context_len=raw_data_config.context_len,
        held_out_chrs=raw_data_config.held_out_chrs,
        train_ratio=raw_data_config.train_ratio,
        bed_path=raw_data_config.bed_path,
        bed_columns=raw_data_config.bed_columns,
        profile_paths=raw_data_config.profile_paths,
        control_paths=raw_data_config.control_paths,
        atac_paths=raw_data_config.atac_paths,
        genome_path=raw_data_config.genome_path,
        chr_refseq=raw_data_config.chr_refseq,
        chr_lengths=raw_data_config.chr_lengths,
        split='all',
        jitter_max=1000,
        reverse_complement_p=0.5,
    )
    data = dataset[0]
    assert data['features'].shape == (20000, 6)
    assert data['profile'].shape == (20000,)
    assert data['control'].shape == (20000,)
