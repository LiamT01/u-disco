import os.path as osp
from typing import Type, TypeVar

from omegaconf import OmegaConf

T = TypeVar('T')


def load_config(
        config_path: str,
        schema: Type[T],
) -> T:
    raw = OmegaConf.load(config_path)
    validated = OmegaConf.merge(schema, raw)
    converted = OmegaConf.to_object(validated)
    return converted


def load_config_composed(
        config_dir: str,
        metaconfig_name: str,
        combined_schema: Type[T],
) -> T:
    metaconfig = OmegaConf.load(osp.join(config_dir, metaconfig_name))
    loaded = {}
    for group, name in metaconfig.items():
        if not name.endswith('.yaml'):
            name = name + '.yaml'
        loaded[group] = OmegaConf.load(osp.join(config_dir, group, name))
    validated = OmegaConf.merge(combined_schema, loaded)
    converted = OmegaConf.to_object(validated)
    return converted
