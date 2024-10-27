import argparse
import os.path as osp

from src.model_dev import train_epochs, eval_exp
from src.types import ExpConfig
from src.utils import load_config_composed, load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    # --------------------- Train ---------------------
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--meta_config', '-c', type=str, help='Meta config file name')

    # --------------------- Eval ---------------------
    eval_parser = subparsers.add_parser('eval', help='Evaluate models')
    eval_parser.add_argument("--exp_dir", '-d', type=str, required=True,
                             help='Path to the experiment directory')
    eval_parser.add_argument('--overwrite_atac_paths', '--atacs', nargs='*', type=str, default=None,
                             help='Overwrite the ATAC paths in the config file')

    args = parser.parse_args()
    if args.cmd == 'train':
        config = load_config_composed('configs', args.meta_config, ExpConfig)
        train_epochs(config)
    elif args.cmd == 'eval':
        config = load_config(osp.join(args.exp_dir, 'config.yaml'), ExpConfig)
        eval_exp(args.exp_dir, config, args.overwrite_atac_paths)
