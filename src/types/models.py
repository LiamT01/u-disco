from abc import ABC, abstractmethod
from typing import TypeVar
from typing import TypedDict, Type

import torch
from torch import nn as nn

from .configs import ExpConfig

T = TypeVar('T')
t_wrapper_model_forward_return = TypedDict("t_wrapper_model_forward_return", {
    "loss": torch.Tensor,
    "reg_loss": torch.Tensor,
    "profile": torch.Tensor,
})


class ModelFromConfigMixin(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls: Type[T], config: ExpConfig) -> T:
        ...


class IBackendModel(nn.Module, ModelFromConfigMixin, ABC):
    ...


class ISeqModel(nn.Module, ModelFromConfigMixin, ABC):
    @abstractmethod
    def run_batch(
            self,
            features: torch.Tensor,
            profile: torch.Tensor,
            control: torch.Tensor | None = None,
            use_prior: bool = False,
            reg_loss_weight: float = 1.,
            return_probs: bool = False,
    ) -> t_wrapper_model_forward_return:
        ...
