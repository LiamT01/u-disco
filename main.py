import argparse
import os
import os.path as osp
import pathlib

import modiscolite

from src.interpret import interpret_model
from src.model_dev import train_epochs, eval_exp
from src.motifs import identify_motifs, annotate_patterns, calc_co_occurrence, calc_preference
from src.types import ExpConfig
from src.utils import load_config_composed, load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    # --------------------- Train ---------------------
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--meta_config', '-c',
                              type=str, required=True,
                              help='Meta config file name')

    # --------------------- Eval ---------------------
    eval_parser = subparsers.add_parser('eval', help='Evaluate models')
    eval_parser.add_argument('--exp_dir', '-d',
                             type=str, required=True,
                             help='Path to the experiment directory')
    eval_parser.add_argument('--overwrite_atac_paths', '--atacs', nargs='*',
                             type=str, default=None,
                             help='Overwrite the ATAC paths in the config file')

    # --------------------- Interpret ---------------------
    interpret_parser = subparsers.add_parser('interpret', help='Interpret a trained model')
    interpret_parser.add_argument('--split', '-s',
                                  type=str, required=True,
                                  help='Dataset split to interpret (train, val, test, all)')
    interpret_parser.add_argument('--exp_dir', '-d',
                                  type=str, required=True,
                                  help='Path to the experiment directory')
    interpret_parser.add_argument('--ckpt_name', '-c',
                                  type=str, required=True,
                                  help='Checkpoint name (e.g. epoch_100.pth)')
    interpret_parser.add_argument('--seed', '-r',
                                  type=int, default=0,
                                  help='Random seed')
    interpret_parser.add_argument('--n_background', '-b',
                                  type=int, default=100,
                                  help='Number of background sequences')
    interpret_parser.add_argument('--n_samples', '-n',
                                  type=int, default=200,
                                  help='Number of samples')
    interpret_parser.add_argument('--batch_size', '--bs',
                                  type=int, default=10,
                                  help='Batch size')

    # --------------------- Motif identification ---------------------
    motif_ident_parser = subparsers.add_parser('identify', help='Identify motifs from interpretation results')
    motif_ident_parser.add_argument('--exp_dir', '-d',
                                    type=str, required=True,
                                    help='Path to the experiment directory')
    motif_ident_parser.add_argument('--split', '-s',
                                    type=str, required=True,
                                    help='Dataset split to interpret (train, val, test, all)')
    motif_ident_parser.add_argument('--max_seqlets', '-n',
                                    type=int, required=True,
                                    help='Maximum number of seqlets per metacluster')
    motif_ident_parser.add_argument('--window_size', '-w',
                                    type=int, required=True,
                                    help='Size of window (in base pairs) surrounding peak center (area outside window will be ignored)')
    motif_ident_parser.add_argument('--n_leiden', '-l',
                                    type=int, default=2,
                                    help='Number of Leiden clusterings to perform with different random seeds')
    motif_ident_parser.add_argument('--verbose', '-v',
                                    action='store_true', default=False,
                                    help='Print verbose output')

    # --------------------- Report ---------------------
    motif_report_parser = subparsers.add_parser('report',
                                                help='Generate motif report, optionally against a motif database')
    motif_report_parser.add_argument('--h5py', '-i',
                                     type=str, required=True,
                                     help='Path to HDF5 file containing the identified motifs.')
    motif_report_parser.add_argument('--output_dir', '-o',
                                     type=str, required=True,
                                     help='Output directory for the report.')
    motif_report_parser.add_argument('--meme_db', '-m',
                                     type=str, default=None,
                                     help='Path to MEME database file containing motifs.')
    motif_report_parser.add_argument('--n_matches', '-n',
                                     type=int, default=3,
                                     help='Number of top TOMTOM matches to include in the report.')

    # --------------------- Annotate ---------------------
    motif_annot_parser = subparsers.add_parser('annotate', help='Annotate patterns and generate motif instances')
    motif_annot_parser.add_argument('--exp_dir', '-d',
                                    type=str, required=True,
                                    help='Path to the experiment directory')
    motif_annot_parser.add_argument('--split', '-s',
                                    type=str, required=True,
                                    help='Dataset split to interpret (train, val, test, all)')
    motif_annot_parser.add_argument('--h5py_path', '-i',
                                    type=str, required=True,
                                    help='Path to HDF5 file containing the identified motifs.')
    motif_annot_parser.add_argument('--annot_yaml_path', '-a',
                                    type=str, required=True,
                                    help='Path to the annotation YAML file.')
    motif_annot_parser.add_argument('--output_dir', '-o',
                                    type=str, required=True,
                                    help='Output directory for the motif instances YAML file.')

    # --------------------- Co-occurrence ---------------------
    co_occurrence_parser = subparsers.add_parser('co_occurrence', help='Calculate co-occurrence between motifs')
    co_occurrence_parser.add_argument('--config_path', '-c',
                                      type=str, required=True,
                                      help='Path to the config file for co-occurrence calculation')
    co_occurrence_parser.add_argument('--annot_yaml_path', '-a',
                                      type=str, required=True,
                                      help='Path to the annotation YAML file.')
    co_occurrence_parser.add_argument('--motif_instances_yaml_path', '-m',
                                      type=str, required=True,
                                      help='Path to the motif instances YAML file.')
    co_occurrence_parser.add_argument('--output_dir', '-o',
                                      type=str, required=True,
                                      help='Output directory for the co-occurrence results.')

    # --------------------- Preference ---------------------
    preference_parser = subparsers.add_parser('preference', help='Calculate preference between motifs')
    preference_parser.add_argument('--config_path', '-c',
                                   type=str, required=True,
                                   help='Path to the config file for preference calculation')
    preference_parser.add_argument('--annot_yaml_path', '-a',
                                   type=str, required=True,
                                   help='Path to the annotation YAML file.')
    preference_parser.add_argument('--motif_instances_yaml_path', '-m',
                                   type=str, required=True,
                                   help='Path to the motif instances YAML file.')
    preference_parser.add_argument('--output_dir', '-o',
                                   type=str, required=True,
                                   help='Output directory for the preference results.')

    args = parser.parse_args()
    if args.cmd == 'train':
        config = load_config_composed('configs', args.meta_config, ExpConfig)
        train_epochs(config)
    elif args.cmd == 'eval':
        config = load_config(osp.join(args.exp_dir, 'config.yaml'), ExpConfig)
        eval_exp(args.exp_dir, config, args.overwrite_atac_paths)
    elif args.cmd == 'interpret':
        interpret_model(
            split=args.split,
            exp_dir=args.exp_dir,
            ckpt_name=args.ckpt_name,
            seed=args.seed,
            n_background=args.n_background,
            n_samples=args.n_samples,
            batch_size=args.batch_size
        )
    elif args.cmd == 'identify':
        ohe_seq_path = osp.join(args.exp_dir, f'interpret-{args.split}', 'ohe_seq.npz')
        seq_shap_path = osp.join(args.exp_dir, f'interpret-{args.split}', 'seq_shap_centered.npz')
        output_dir = osp.join(args.exp_dir, f'motifs-{args.split}')

        identify_motifs(
            ohe_seq_path=ohe_seq_path,
            attr_path=seq_shap_path,
            output_dir=output_dir,
            window_size=args.window_size,
            max_seqlets=args.max_seqlets,
            n_leiden=args.n_leiden,
            verbose=args.verbose
        )
    elif args.cmd == 'report':
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Writing report to {args.output_dir}', flush=True)

        modiscolite.report.report_motifs(
            pathlib.Path(args.h5py),
            pathlib.Path(args.output_dir),
            img_path_suffix=pathlib.Path('./'),
            meme_motif_db=args.meme_db,
            is_writing_tomtom_matrix=False,
            top_n_matches=args.n_matches,
        )
    elif args.cmd == 'annotate':
        annotate_patterns(
            exp_dir=args.exp_dir,
            split=args.split,
            h5py_path=args.h5py_path,
            annot_yaml_path=args.annot_yaml_path,
            output_dir=args.output_dir,
        )
    elif args.cmd == 'co_occurrence':
        calc_co_occurrence(
            config_path=args.config_path,
            annot_yaml_path=args.annot_yaml_path,
            motif_instances_yaml_path=args.motif_instances_yaml_path,
            output_dir=args.output_dir,
        )
    elif args.cmd == 'preference':
        calc_preference(
            config_path=args.config_path,
            annot_yaml_path=args.annot_yaml_path,
            motif_instances_yaml_path=args.motif_instances_yaml_path,
            output_dir=args.output_dir,
        )
