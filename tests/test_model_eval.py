import os.path as osp
from glob import glob

from src.model_dev import train_epochs, eval_exp
from src.types import ExpConfig
from src.utils import load_config_composed, load_config


def _train_exp(
        config_dir: str,
        metaconfig_name: str,
) -> str:
    train_config = load_config_composed(config_dir, metaconfig_name, ExpConfig)
    train_config.model_dev.n_epochs = 1
    exp_dir = train_epochs(train_config)
    return exp_dir


def _eval_exp(exp_dir: str):
    config = load_config(osp.join(exp_dir, 'config.yaml'), ExpConfig)
    eval_exp(exp_dir, config, None)


def test_train_eval():
    for meta_config_path in glob('configs/*.yaml'):
        config_dir, metaconfig_name = meta_config_path.split('/')
        exp_dir = _train_exp(config_dir, metaconfig_name)
        _eval_exp(exp_dir)
