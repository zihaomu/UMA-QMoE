"""PyTorch correctness and selector-fidelity oracles for Faster Flash Decoding."""

from __future__ import annotations

import math
from typing import Any

from ..contracts import ContractError
from .quantization import QuantizedKeyBlocks, dequantize_key_blocks, quantize_key_blocks


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ContractError("FFD oracle requires PyTorch") from exc
    return torch


def _normalize_query(query: Any) -> Any:
    if query.ndim == 4:
        if query.shape[1] == 1:
            query = query[:, 0]
        elif query.shape[2] == 1:
            query = query[:, :, 0]
        else:
            raise ContractError("FFD decode query must contain exactly one token")
    if query.ndim != 3:
        raise ContractError("FFD query must have shape [B, H, D] or one-token 4D")
    return query


def exact_attention_scores(query: Any, keys: Any, *, scale: float | None = None) -> Any:
    """Return FP32 scores as ``[batch, query_heads, tokens]``."""

    torch = _torch()
    query = _normalize_query(query)
    if keys.ndim != 4:
        raise ContractError("FFD keys must have shape [B, T, KVH, D]")
    batch, tokens, kv_heads, head_dim = keys.shape
    if query.shape[0] != batch or query.shape[-1] != head_dim:
        raise ContractError("FFD query/key batch or head dimension mismatch")
    query_heads = query.shape[1]
    if query_heads % kv_heads:
        raise ContractError("FFD query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    head_to_kv = torch.arange(query_heads, device=query.device) // groups
    keys_by_head = keys[:, :, head_to_kv, :].permute(0, 2, 1, 3).float()
    factor = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("bhd,bhtd->bht", query.float(), keys_by_head) * factor
    if scores.shape != (batch, query_heads, tokens):
        raise ContractError("FFD score oracle produced an unexpected shape")
    return scores


def dense_attention(
    query: Any, keys: Any, values: Any, *, scale: float | None = None
) -> Any:
    """FP32 decode attention reference returning ``[B, H, V]``."""

    torch = _torch()
    scores = exact_attention_scores(query, keys, scale=scale)
    if values.ndim != 4 or values.shape[:3] != keys.shape[:3]:
        raise ContractError("FFD values must have shape [B, T, KVH, V]")
    query_heads = scores.shape[1]
    kv_heads = values.shape[2]
    groups = query_heads // kv_heads
    head_to_kv = torch.arange(query_heads, device=values.device) // groups
    values_by_head = values[:, :, head_to_kv, :].permute(0, 2, 1, 3).float()
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,bhtv->bhv", probabilities, values_by_head)


def pseudo_max(scores: Any, *, sink_tokens: int, local_tokens: int) -> Any:
    """Maximum over the union of fixed sink and most-recent local tokens."""

    if scores.ndim != 3 or scores.shape[-1] <= 0:
        raise ContractError("FFD scores must have shape [B, H, T] with T > 0")
    token_count = scores.shape[-1]
    if sink_tokens <= 0 or local_tokens <= 0:
        raise ContractError("FFD sink/local windows must be positive")
    sink_end = min(sink_tokens, token_count)
    local_start = max(0, token_count - local_tokens)
    sink_max = scores[..., :sink_end].amax(dim=-1)
    local_max = scores[..., local_start:].amax(dim=-1)
    return sink_max.maximum(local_max)


def block_maxima(scores: Any, block_size: int) -> Any:
    """Reduce token scores to padded block maxima."""

    torch = _torch()
    if scores.ndim != 3:
        raise ContractError("FFD scores must have shape [B, H, T]")
    if block_size <= 0:
        raise ContractError("FFD block_size must be positive")
    token_count = scores.shape[-1]
    block_count = math.ceil(token_count / block_size)
    padded_tokens = block_count * block_size
    if padded_tokens != token_count:
        scores = torch.nn.functional.pad(
            scores, (0, padded_tokens - token_count), value=float("-inf")
        )
    return scores.reshape(*scores.shape[:-1], block_count, block_size).amax(dim=-1)


def expand_block_mask(block_mask: Any, *, block_size: int, token_count: int) -> Any:
    if block_mask.ndim != 3:
        raise ContractError("FFD block mask must have shape [B, H, blocks]")
    return block_mask.repeat_interleave(block_size, dim=-1)[..., :token_count]


def top_delta_block_mask(scores: Any, threshold: Any, *, block_size: int) -> Any:
    if threshold.shape != scores.shape[:-1]:
        raise ContractError("FFD threshold shape must match score batch/head axes")
    return block_maxima(scores, block_size) >= threshold[..., None]


def topk_matched_block_mask(scores: Any, block_counts: Any, *, block_size: int) -> Any:
    """Select exact-score blocks with a per-head count matched to another policy."""

    torch = _torch()
    maxima = block_maxima(scores, block_size)
    if block_counts.shape != maxima.shape[:-1]:
        raise ContractError("FFD top-k block counts must have shape [B, H]")
    result = torch.zeros_like(maxima, dtype=torch.bool)
    for count in range(1, maxima.shape[-1] + 1):
        selector = block_counts == count
        if bool(selector.any().item()):
            indices = torch.topk(maxima, k=count, dim=-1).indices
            candidate = torch.zeros_like(result)
            candidate.scatter_(-1, indices, True)
            result = torch.where(selector[..., None], candidate, result)
    return result


def _masked_attention(scores: Any, values: Any, token_mask: Any) -> Any:
    torch = _torch()
    query_heads = scores.shape[1]
    kv_heads = values.shape[2]
    groups = query_heads // kv_heads
    head_to_kv = torch.arange(query_heads, device=values.device) // groups
    values_by_head = values[:, :, head_to_kv, :].permute(0, 2, 1, 3).float()
    masked_scores = scores.masked_fill(~token_mask, float("-inf"))
    probabilities = torch.softmax(masked_scores, dim=-1)
    return torch.einsum("bht,bhtv->bhv", probabilities, values_by_head)


def _summary(values: Any) -> dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "p01": float(flat.quantile(0.01).item()),
        "p50": float(flat.quantile(0.50).item()),
        "p95": float(flat.quantile(0.95).item()),
        "p99": float(flat.quantile(0.99).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def selector_fidelity_report(
    query: Any,
    keys: Any,
    values: Any,
    *,
    delta: float,
    block_size: int,
    sink_tokens: int,
    local_tokens: int,
    key_bits: int = 2,
) -> dict[str, Any]:
    """Compare ideal, exact pseudo-max, and Q2 pseudo-max top-delta selectors."""

    torch = _torch()
    query = _normalize_query(query)
    if keys.shape[1] < block_size:
        raise ContractError("FFD selector report needs at least one complete block")
    exact_scores = exact_attention_scores(query, keys)
    token_count = keys.shape[1]
    full_tokens = token_count // block_size * block_size
    quantized = quantize_key_blocks(
        keys[:, :full_tokens],
        block_size=block_size,
        bits=key_bits,
        residual_dtype="none",
    )
    approximate_full = exact_attention_scores(
        query, dequantize_key_blocks(quantized, include_residual=False)
    )
    if full_tokens < token_count:
        approximate_scores = torch.cat(
            (approximate_full, exact_scores[..., full_tokens:]), dim=-1
        )
    else:
        approximate_scores = approximate_full

    global_max = exact_scores.amax(dim=-1)
    exact_pseudo = pseudo_max(
        exact_scores, sink_tokens=sink_tokens, local_tokens=local_tokens
    )
    q2_pseudo = pseudo_max(
        approximate_scores, sink_tokens=sink_tokens, local_tokens=local_tokens
    )
    ideal_token_mask = exact_scores >= global_max[..., None] - float(delta)
    ideal_block_mask = block_maxima(ideal_token_mask.float(), block_size) > 0
    exact_block_mask = top_delta_block_mask(
        exact_scores, exact_pseudo - float(delta), block_size=block_size
    )
    q2_block_mask = top_delta_block_mask(
        approximate_scores, q2_pseudo - float(delta), block_size=block_size
    )
    # The high-precision tail is always included by the runtime.
    if full_tokens < token_count:
        q2_block_mask[..., -1] = True
        exact_block_mask[..., -1] = True

    q2_token_mask = expand_block_mask(
        q2_block_mask, block_size=block_size, token_count=token_count
    )
    exact_token_mask = expand_block_mask(
        exact_block_mask, block_size=block_size, token_count=token_count
    )
    ideal_count = ideal_block_mask.sum(dim=-1)
    true_positive = (q2_block_mask & ideal_block_mask).sum(dim=-1)
    selected_count = q2_block_mask.sum(dim=-1)
    block_recall = true_positive.float() / ideal_count.clamp_min(1).float()
    block_precision = true_positive.float() / selected_count.clamp_min(1).float()
    false_negative = ideal_token_mask & ~q2_token_mask
    false_negative_rate = (
        false_negative.sum(dim=-1).float()
        / ideal_token_mask.sum(dim=-1).clamp_min(1).float()
    )
    probabilities = torch.softmax(exact_scores, dim=-1)
    selected_mass = (probabilities * q2_token_mask).sum(dim=-1)
    exact_selected_mass = (probabilities * exact_token_mask).sum(dim=-1)
    keep_ratio = q2_block_mask.float().mean(dim=-1)

    dense_output = dense_attention(query, keys, values)
    sparse_output = _masked_attention(exact_scores, values, q2_token_mask)
    difference = sparse_output - dense_output
    dense_lse = torch.logsumexp(exact_scores, dim=-1)
    sparse_lse = torch.logsumexp(
        exact_scores.masked_fill(~q2_token_mask, float("-inf")), dim=-1
    )
    lse_error = (sparse_lse - dense_lse).abs()
    relative_error = difference.norm(dim=-1) / dense_output.norm(dim=-1).clamp_min(
        1e-12
    )
    cosine = torch.nn.functional.cosine_similarity(sparse_output, dense_output, dim=-1)

    return {
        "schema_version": 1,
        "delta": float(delta),
        "key_bits": key_bits,
        "block_size": block_size,
        "sink_tokens": sink_tokens,
        "local_tokens": local_tokens,
        "shape": {
            "batch": int(keys.shape[0]),
            "tokens": int(token_count),
            "query_heads": int(query.shape[1]),
            "kv_heads": int(keys.shape[2]),
            "head_dim": int(keys.shape[3]),
        },
        "pseudo_max_gap": _summary(global_max - exact_pseudo),
        "q2_pseudo_max_gap": _summary(global_max - q2_pseudo),
        "block_recall": _summary(block_recall),
        "block_precision": _summary(block_precision),
        "salient_token_false_negative_rate": _summary(false_negative_rate),
        "selected_attention_mass": _summary(selected_mass),
        "exact_selector_attention_mass": _summary(exact_selected_mass),
        "keep_ratio": _summary(keep_ratio),
        "attention_output_cosine": _summary(cosine),
        "attention_output_relative_error": _summary(relative_error),
        "attention_output_max_abs_error": float(difference.abs().max().item()),
        "lse_absolute_error": _summary(lse_error),
        "per_head": {
            "block_recall": block_recall.detach().cpu().tolist(),
            "block_precision": block_precision.detach().cpu().tolist(),
            "false_negative_rate": false_negative_rate.detach().cpu().tolist(),
            "selected_mass": selected_mass.detach().cpu().tolist(),
            "keep_ratio": keep_ratio.detach().cpu().tolist(),
        },
    }


def q2_sparse_attention(
    query: Any,
    quantized: QuantizedKeyBlocks,
    values: Any,
    *,
    delta: float,
    sink_tokens: int,
    local_tokens: int,
    tail_keys: Any | None = None,
    tail_values: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Slow reference implementation of complete Q2+residual sparse decode."""

    torch = _torch()
    approximate_keys = dequantize_key_blocks(quantized, include_residual=False)
    refined_keys = dequantize_key_blocks(
        quantized, include_residual=quantized.residual is not None
    )
    approximate_scores = exact_attention_scores(query, approximate_keys)
    threshold_scores = approximate_scores
    all_values = values
    all_refined_keys = refined_keys
    tail_count = 0
    if tail_keys is not None:
        if tail_values is None or tail_keys.shape[:3] != tail_values.shape[:3]:
            raise ContractError("FFD tail K/V shapes do not match")
        tail_count = int(tail_keys.shape[1])
        tail_scores = exact_attention_scores(query, tail_keys)
        threshold_scores = torch.cat((approximate_scores, tail_scores), dim=-1)
        all_refined_keys = torch.cat((refined_keys, tail_keys), dim=1)
        all_values = torch.cat((values, tail_values), dim=1)
    threshold = pseudo_max(
        threshold_scores, sink_tokens=sink_tokens, local_tokens=local_tokens
    ) - float(delta)
    full_block_mask = top_delta_block_mask(
        approximate_scores, threshold, block_size=quantized.block_size
    )
    full_token_mask = expand_block_mask(
        full_block_mask,
        block_size=quantized.block_size,
        token_count=quantized.token_count,
    )
    if tail_count:
        tail_mask = torch.ones(
            (*full_token_mask.shape[:-1], tail_count),
            dtype=torch.bool,
            device=full_token_mask.device,
        )
        token_mask = torch.cat((full_token_mask, tail_mask), dim=-1)
    else:
        token_mask = full_token_mask
    refined_scores = exact_attention_scores(query, all_refined_keys)
    output = _masked_attention(refined_scores, all_values, token_mask)
    return output, {
        "selected_blocks": int(full_block_mask.sum().item()),
        "total_blocks": int(full_block_mask.numel()),
        "keep_ratio": float(full_block_mask.float().mean().item()),
        "tail_tokens": tail_count,
    }


__all__ = [
    "block_maxima",
    "dense_attention",
    "exact_attention_scores",
    "expand_block_mask",
    "pseudo_max",
    "q2_sparse_attention",
    "selector_fidelity_report",
    "top_delta_block_mask",
    "topk_matched_block_mask",
]
