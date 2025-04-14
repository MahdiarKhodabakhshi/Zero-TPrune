from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


use_s_stage = True
sim_feature = "Key"
partition = "Seq-U"
sim_metric = "Cos"
use_threshold = False
save_mask = False


PartitionT = Literal["Seq-U", "Seq-I", "Rand", "Alt", "Full"]
SimMetricT = Literal["Dot", "Cos", "Man", "Euc", "Mink3", "Mink4", "Mink5", "MinkInf"]


@dataclass(frozen=True)
class SStageConfig:
    tau_imp: float = 0.1 
    metric: SimMetricT = "Cos"
    partition: PartitionT = "Seq-U"
    use_threshold: bool = False
    save_mask: bool = False

AugMethodT = Literal["weight", "norm"]
AltMethodT = Literal["ave", "rand"]

@dataclass(frozen=True)
class IStageConfig:
    use_WPR: bool = True
    use_EIR: bool = True
    aug_CLS: bool = True
    aug_method: AugMethodT = "weight"
    alt_method: AltMethodT = "ave"

    iters: int = 1
    d: float = 0.0
    var_filter: int = 0


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


def i_stage_importance(
    attention_score: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    *,
    tau_imp: float = 0.1,
    i_cfg: Optional[IStageConfig] = None,
    v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    i_cfg = i_cfg or IStageConfig()

    B, H, N, _ = attention_score.shape
    attn = (attention_score / tau_imp).exp()
    if token_mask is not None:
        if token_mask.dim() == 2:
            attn = attn * token_mask[:, None, None, :]
        else:
            attn = attn * token_mask

    M = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if i_cfg.use_WPR:
        if i_cfg.aug_CLS:
            if i_cfg.aug_method == "weight":
                alpha = float(N) / 2.0
                init = torch.ones(B, H, 1, N, device=M.device)
                init[:, :, :, 0] = 1.0 + alpha
                dist = init / (N + alpha)
            elif i_cfg.aug_method == "norm":
                if v is None:
                    alpha = float(N) / 2.0
                    init = torch.ones(B, H, 1, N, device=M.device)
                    init[:, :, :, 0] = 1.0 + alpha
                    dist = init / (N + alpha)
                else:
                    init = attention_score[:, :, 0, :].unsqueeze(-2) * torch.norm(v, dim=-1).unsqueeze(-2)
                    alpha = init.sum(dim=-1)
                    init[:, :, :, 0] = init[:, :, :, 0] + alpha
                    dist = init / init.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                raise ValueError("Invalid aug_method")
        else:
            dist = torch.ones(B, H, 1, N, device=M.device) / N

        if i_cfg.iters < 1:
            raise ValueError("iters must be >= 1")
        d = float(i_cfg.d)
        for _ in range(i_cfg.iters):
            dist = (dist @ M) * (1.0 - d) + (d / N)

        if i_cfg.var_filter == 1:
            dist_v = dist.squeeze(2)

        dist = dist.squeeze(2)
        if i_cfg.use_EIR:
            dist = torch.mean(dist.pow(2), dim=1).sqrt()
        else:
            dist = torch.mean(dist, dim=1)
    else:
        if i_cfg.alt_method == "ave":
            base = torch.mean(M, dim=2)
            dist = torch.mean(base.pow(2), dim=1).sqrt() if i_cfg.use_EIR else torch.mean(base, dim=1)
        elif i_cfg.alt_method == "rand":
            dist = torch.rand(B, N, device=M.device)
        else:
            raise ValueError("Invalid alt_method")

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


def s_stage_step(
    attention_score: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    *,
    cfg: Optional[SStageConfig] = None,
    k_keep: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = cfg or SStageConfig(
        tau_imp=0.1,
        metric=sim_metric,
        partition=partition,
        use_threshold=use_threshold,
        save_mask=save_mask,
    )

    B, H, N, _ = attention_score.shape
    device = attention_score.device

    importance = s_stage_importance(attention_score, token_mask, tau_imp=cfg.tau_imp)

    order = _partition_order(B, N - 1, cfg.partition, device)

    next_mask = s_stage_select(
        importance=importance,
        order=order,
        k_keep=k_keep,
        threshold=threshold,
        use_threshold=cfg.use_threshold,
    )

    if cfg.save_mask:
        torch.save(next_mask.detach().cpu(), "mask.pt")

    return importance, next_mask

if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, N = 2, 4, 9
    logits = torch.randn(B, H, N, N)

    imp, nxt = s_stage_step(
        attention_score=logits,
        token_mask=None,
        cfg=SStageConfig(
            tau_imp=0.1,
            metric="Cos",
            partition="Seq-U",
            use_threshold=False,
            save_mask=False,
        ),
        k_keep=4,
    )

    print("importance shape:", imp.shape)
    print("next_mask shape :", nxt.shape)
    print("kept per sample :", nxt.sum(dim=1).tolist())