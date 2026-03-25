"""PU learning loss functions."""

import torch
import torch.nn.functional as F


def nnpu_loss(
    logits_p: torch.Tensor,
    logits_u: torch.Tensor,
    prior: float,
) -> torch.Tensor:
    """Non-negative PU (nnPU) loss from Kiryo et al. (2017).

    The unbiased PU risk is:
        R = prior * R_p_pos + max(0, R_u_neg - prior * R_p_neg)

    Where:
        R_p_pos = E_P[loss(f(x), 1)]   -- positive risk on P
        R_p_neg = E_P[loss(f(x), 0)]   -- negative risk on P
        R_u_neg = E_U[loss(f(x), 0)]   -- negative risk on U

    The max(0, ...) is the non-negative correction that prevents the
    negative risk term from going below zero (which destabilizes training).

    Args:
        logits_p: Raw logits for P (labeled positive) samples. Shape (n_p,).
        logits_u: Raw logits for U (unlabeled) samples. Shape (n_u,).
        prior: Estimated class prior (fraction of positives in the full dataset).

    Returns:
        Scalar loss tensor.
    """
    # Positive risk: how badly do we classify P samples as positive?
    loss_p_pos = F.binary_cross_entropy_with_logits(
        logits_p, torch.ones_like(logits_p), reduction="mean"
    )

    # Negative risk on P: how badly do we classify P samples as negative?
    loss_p_neg = F.binary_cross_entropy_with_logits(
        logits_p, torch.zeros_like(logits_p), reduction="mean"
    )

    # Negative risk on U: how badly do we classify U samples as negative?
    loss_u_neg = F.binary_cross_entropy_with_logits(
        logits_u, torch.zeros_like(logits_u), reduction="mean"
    )

    # Unbiased PU risk with non-negative correction
    positive_risk = prior * loss_p_pos
    negative_risk = loss_u_neg - prior * loss_p_neg

    # nnPU: clip negative risk to zero
    if negative_risk.item() < 0:
        return positive_risk - negative_risk  # gradient ascent on negative part
    return positive_risk + negative_risk
