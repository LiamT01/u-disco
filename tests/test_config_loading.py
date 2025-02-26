from src.types import ExpConfig
from src.utils import load_config_composed


def test_load_config():
    load_config_composed('configs', 'u-disco_seq_only.yaml', ExpConfig)
