import importlib

import torch

from src.metrics import profile_logits_to_log_probs, multinomial_log_probs, fourier_att_prior_loss
from src.types import ExpConfig, IBackendModel, ISeqModel, t_wrapper_model_forward_return


class DNASeqModel(ISeqModel):
    vocab_size: int
    n_epi: int
    noise_std: float
    backend: IBackendModel

    def __init__(
            self,
            vocab_size: int,
            n_epi: int,
            backend: IBackendModel,
            noise_std: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_epi = n_epi
        self.noise_std = noise_std
        self.backend = backend

    def forward(
            self,
            data: torch.Tensor,
            control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training and self.n_epi > 0:
            # Avoid in-place operation
            data = data + torch.cat(
                [
                    torch.zeros_like(data[..., :self.vocab_size]),
                    torch.randn_like(data[..., self.vocab_size:]) * self.noise_std
                ],
                dim=-1,
            )
        return self.backend(data, control)

    @staticmethod
    def prof_loss(
            logit_pred_profs: torch.Tensor,
            true_profs: torch.Tensor,
    ) -> torch.Tensor:
        true_counts = torch.sum(true_profs, dim=-1)
        log_pred_profs = profile_logits_to_log_probs(logit_pred_profs, axis=-1)
        neg_log_likelihood = -multinomial_log_probs(log_pred_profs, true_counts, true_profs)
        prof_loss = torch.mean(neg_log_likelihood)
        return prof_loss

    @staticmethod
    def reg_loss(input_grads: torch.Tensor) -> torch.Tensor:
        return fourier_att_prior_loss(
            torch.ones(input_grads.size(0)).to(input_grads.device),
            input_grads,
            freq_limit=3000,
            limit_softness=0.2,
            att_prior_grad_smooth_sigma=3,
        )

    def run_batch(
            self,
            features: torch.Tensor,
            profile: torch.Tensor,
            control: torch.Tensor | None = None,
            use_prior: bool = False,
            reg_loss_weight: float = 1.,
            return_probs: bool = False,
    ) -> t_wrapper_model_forward_return:
        if use_prior:
            features.requires_grad = True
            logit_pred_vals = self(features, control)
            # Compute the gradients of the output with respect to the input
            input_grads, = torch.autograd.grad(
                logit_pred_vals, features,
                grad_outputs=torch.ones(logit_pred_vals.size()).to(features.device),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
            )
            input_grads = input_grads * features  # Gradient * input
            features.requires_grad = False  # Reset gradient required
            loss = self.prof_loss(logit_pred_vals, profile)
            reg_loss = self.reg_loss(input_grads) * reg_loss_weight
        else:
            logit_pred_vals = self(features, control)
            loss = self.prof_loss(logit_pred_vals, profile)
            reg_loss = torch.tensor(0., device=features.device)

        if return_probs:
            return {
                "loss": loss,
                "reg_loss": reg_loss,
                "profile": torch.softmax(logit_pred_vals, dim=-1),
            }
        else:
            return {
                "loss": loss,
                "reg_loss": reg_loss,
                "profile": logit_pred_vals,
            }

    @classmethod
    def from_config(
            cls,
            config: ExpConfig,
    ) -> 'DNASeqModel':
        module = importlib.import_module(config.model_dev.backend_model_module)
        model_class = getattr(module, config.model_dev.backend_model_class)
        backend = model_class.from_config(config)
        return cls(
            vocab_size=config.model_dev.vocab_size,
            n_epi=config.model_dev.n_epi,
            backend=backend,
            noise_std=config.model_dev.noise_std,
        )
