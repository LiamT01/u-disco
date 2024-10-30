import torch
from torch import nn

from src.metrics import profile_logits_to_log_probs
from src.types import ISeqModel


class WrapperProfileModel(nn.Module):
    def __init__(self, inner_model: ISeqModel):
        """
        Takes a profile model and constructs wrapper model around it. This model
        takes in the same inputs (i.e. input tensor of shape B x I x 4 and
        perhaps a set of control profiles of shape B x (T or 1) x O x S). The
        model will return an output of B x 1, which is the profile logits
        (weighted), aggregated to a scalar for each input.
        :param inner_model: a trained model
        """
        super().__init__()
        self.inner_model = inner_model

    def forward(
            self,
            input_seqs: torch.Tensor,
            control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Run through inner model, disregarding the predicted counts
        logit_pred_profs = self.inner_model(input_seqs, control)

        # As with the computation of the gradients, instead of explaining the
        # logits, explain the mean-normalized logits, weighted by the final
        # probabilities after passing through the softmax; this exponentially
        # increases the weight for high-probability positions, and exponentially
        # reduces the weight for low-probability positions, resulting in a
        # cleaner signal

        # Subtract mean along output profile dimension; this wouldn't change
        # softmax probabilities, but normalizes the magnitude of the logits
        norm_logit_pred_profs = logit_pred_profs - logit_pred_profs.mean(dim=-1, keepdim=True)

        # Weight by post-softmax probabilities, but detach it from the graph to
        # avoid explaining those
        pred_prof_probs = profile_logits_to_log_probs(
            logit_pred_profs
        ).detach()
        weighted_norm_logits = norm_logit_pred_profs * pred_prof_probs

        # DeepSHAP requires the shape to be B x 1
        prof_sum = torch.sum(weighted_norm_logits, dim=-1, keepdim=True)
        return prof_sum
