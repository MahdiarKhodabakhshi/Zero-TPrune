from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


use_s_stage = True  # whether to use S-stage for similarity-based pruning
sim_feature = "Key"  # "Key", "Query", "Value", "X_pre", "X" (placeholder here)
partition = "Seq-U"  # "Seq-U", "Seq-I", "Rand", "Alt", "Full"
sim_metric = "Cos"   # "Dot", "Cos", "Man", "Euc", "Mink3", "Mink4", "Mink5", "MinkInf"
use_threshold = False  # if True, use threshold-based pruning instead of top-k
save_mask = False      # save the binary keep-mask for each pruning layer (mask.pt)


PartitionT = Literal["Seq-U", "Seq-I", "Rand", "Alt", "Full"]
SimMetricT = Literal["Dot", "Cos", "Man", "Euc", "Mink3", "Mink4", "Mink5", "MinkInf"]


@dataclass(frozen=True)
class SStageConfig:
    tau_imp: float = 0.1 
    metric: SimMetricT = "Cos"
    partition: PartitionT = "Seq-U"
    use_threshold: bool = False
    save_mask: bool = False


def _row_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def s_stage_importance(
    attention_score: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    tau_imp: float = 0.1,
) -> torch.Tensor:
    B, H, N, _ = attention_score.shape

    attn = (attention_score / tau_imp).exp()
    if token_mask is not None:
        if token_mask.dim() == 2:
            attn = attn * token_mask[:, None, None, :] 
        else:
            attn = attn * token_mask

    M = _row_normalize(attn)

    M_head_mean = M.mean(dim=1) 
    dist = M_head_mean.mean(dim=1)

    return dist[:, 1:]


def _partition_order(
    B: int, n_tokens: int, mode: PartitionT, device: torch.device
) -> torch.Tensor:
    base = torch.arange(1, n_tokens + 1, device=device)

    if mode == "Full":
        order = base.expand(B, -1)
    elif mode == "Seq-U":
        order = base.expand(B, -1)
    elif mode == "Seq-I":
        order = base.flip(0).expand(B, -1)
    elif mode == "Rand":
        order = torch.stack([base[torch.randperm(n_tokens, device=device)] for _ in range(B)], dim=0)
    elif mode == "Alt":
        front = base[: (n_tokens + 1) // 2]
        back = base[(n_tokens + 1) // 2 :].flip(0)
        interleaved = torch.empty_like(base)
        interleaved[0::2] = front
        interleaved[1::2] = back
        order = interleaved.expand(B, -1)
    else:
        raise ValueError(f"Unknown partition mode: {mode}")
    return order


def s_stage_select(
    importance: torch.Tensor,
    order: torch.Tensor,
    k_keep: Optional[int] = None,
    threshold: Optional[float] = None,
    use_threshold: bool = False,
) -> torch.Tensor:
    B, n = importance.shape
    device = importance.device
    next_mask = torch.zeros(B, n, device=device)

    if use_threshold:
        if threshold is None:
            raise ValueError("use_threshold=True requires a numeric 'threshold'.")
        keep = (importance >= threshold).float()
        idx_local = (order - 1).long()
        keep_reordered = keep.gather(dim=1, index=idx_local)
        next_mask.scatter_(dim=1, index=idx_local, src=keep_reordered)
    else:
        if k_keep is None:
            raise ValueError("Top-k selection requires 'k_keep'.")
        topk_vals, topk_idx = torch.topk(importance, k=min(k_keep, n), dim=1, largest=True, sorted=True)
        keep_local = torch.zeros_like(next_mask)
        keep_local.scatter_(1, topk_idx, 1.0)

        idx_local = (order - 1).long()
        prefer = torch.zeros_like(next_mask)
        prefer.scatter_(1, idx_local[:, : min(k_keep, n)], 1.0)
        next_mask = torch.minimum(keep_local + prefer, torch.ones_like(keep_local))

        overfull = next_mask.sum(dim=1) > k_keep
        if overfull.any():
            for b in torch.nonzero(overfull).flatten():
                cand = torch.nonzero(next_mask[b] > 0.5).flatten()
                cand_imp = importance[b, cand]
                _, sel = torch.topk(cand_imp, k=k_keep, largest=True, sorted=False)
                keep_idx = cand[sel]
                row = torch.zeros_like(next_mask[b])
                row[keep_idx] = 1.0
                next_mask[b] = row

    return next_mask