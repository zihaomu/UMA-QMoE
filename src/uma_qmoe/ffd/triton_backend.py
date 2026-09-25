"""Project-owned fused Q2 selector-computer Triton backend for gfx1151.

The implementation is intentionally specialized to the frozen Qwen MHA shape:
batch 1, 16 query/KV heads, head/value dimension 128.  Each program scans one
Q2 block and, only when its top-delta predicate passes, loads the residual and
Value block.  No index list or dequantized Key tensor is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from ..contracts import ContractError
from .backend import BackendCapability, _AuditedBackend, observed_amd_platform
from .cache import FFDLayerCache
from .policy import FFDPolicy

try:  # Triton is available only in the target runtime.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - core CI has no Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _q2_block_scores(
        query,
        packed_key,
        scales,
        batch_index,
        head_index,
        block_index,
        query_stride_b,
        query_stride_h,
        packed_stride_b,
        packed_stride_t,
        packed_stride_h,
        scale_stride_b,
        scale_stride_block,
        scale_stride_h,
        attention_scale,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PACKED_DIM: tl.constexpr,
    ):
        token_offsets = tl.arange(0, BLOCK_SIZE)[:, None]
        dim_offsets = tl.arange(0, HEAD_DIM)[None, :]
        query_values = tl.load(
            query
            + batch_index * query_stride_b
            + head_index * query_stride_h
            + dim_offsets
        ).to(tl.float32)
        packed_offsets = dim_offsets // 4
        shifts = (dim_offsets % 4) * 2
        token_index = block_index * BLOCK_SIZE + token_offsets
        packed_values = tl.load(
            packed_key
            + batch_index * packed_stride_b
            + token_index * packed_stride_t
            + head_index * packed_stride_h
            + packed_offsets
        ).to(tl.int32)
        codes = ((packed_values >> shifts) & 3).to(tl.float32)
        scale_values = tl.load(
            scales
            + batch_index * scale_stride_b
            + block_index * scale_stride_block
            + head_index * scale_stride_h
            + dim_offsets
        ).to(tl.float32)
        dequantized = (codes - 1.5) * scale_values
        return tl.sum(dequantized * query_values, axis=1) * attention_scale

    @triton.jit(do_not_specialize=("full_blocks", "tail_len"))
    def _ffd_threshold_kernel(
        query,
        packed_key,
        scales,
        tail_key,
        threshold,
        full_blocks,
        tail_len,
        query_stride_b,
        query_stride_h,
        packed_stride_b,
        packed_stride_t,
        packed_stride_h,
        scale_stride_b,
        scale_stride_block,
        scale_stride_h,
        tail_stride_b,
        tail_stride_t,
        tail_stride_h,
        attention_scale,
        delta,
        NUM_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PACKED_DIM: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch_index = program // NUM_HEADS
        head_index = program % NUM_HEADS
        maximum = float("-inf")
        if full_blocks > 0:
            first_scores = _q2_block_scores(
                query,
                packed_key,
                scales,
                batch_index,
                head_index,
                0,
                query_stride_b,
                query_stride_h,
                packed_stride_b,
                packed_stride_t,
                packed_stride_h,
                scale_stride_b,
                scale_stride_block,
                scale_stride_h,
                attention_scale,
                BLOCK_SIZE=BLOCK_SIZE,
                HEAD_DIM=HEAD_DIM,
                PACKED_DIM=PACKED_DIM,
            )
            maximum = tl.max(first_scores, axis=0)
            if full_blocks > 1:
                last_scores = _q2_block_scores(
                    query,
                    packed_key,
                    scales,
                    batch_index,
                    head_index,
                    full_blocks - 1,
                    query_stride_b,
                    query_stride_h,
                    packed_stride_b,
                    packed_stride_t,
                    packed_stride_h,
                    scale_stride_b,
                    scale_stride_block,
                    scale_stride_h,
                    attention_scale,
                    BLOCK_SIZE=BLOCK_SIZE,
                    HEAD_DIM=HEAD_DIM,
                    PACKED_DIM=PACKED_DIM,
                )
                # If a high-precision tail exists, only the suffix required to
                # complete the fixed local window participates in pseudo-max.
                last_scores = tl.where(
                    tl.arange(0, BLOCK_SIZE) >= tail_len,
                    last_scores,
                    float("-inf"),
                )
                maximum = tl.maximum(maximum, tl.max(last_scores, axis=0))
        if tail_len > 0:
            token_offsets = tl.arange(0, BLOCK_SIZE)[:, None]
            dim_offsets = tl.arange(0, HEAD_DIM)[None, :]
            query_values = tl.load(
                query
                + batch_index * query_stride_b
                + head_index * query_stride_h
                + dim_offsets
            ).to(tl.float32)
            tail_values = tl.load(
                tail_key
                + batch_index * tail_stride_b
                + token_offsets * tail_stride_t
                + head_index * tail_stride_h
                + dim_offsets,
                mask=token_offsets < tail_len,
                other=0.0,
            ).to(tl.float32)
            tail_scores = tl.sum(tail_values * query_values, axis=1) * attention_scale
            tail_scores = tl.where(
                tl.arange(0, BLOCK_SIZE) < tail_len, tail_scores, float("-inf")
            )
            maximum = tl.maximum(maximum, tl.max(tail_scores, axis=0))
        tl.store(threshold + program, maximum - delta)

    @triton.jit(do_not_specialize=("full_blocks",))
    def _ffd_sparse_block_kernel(
        query,
        packed_key,
        scales,
        residual,
        values,
        threshold,
        partial_output,
        partial_ml,
        kept,
        query_stride_b,
        query_stride_h,
        packed_stride_b,
        packed_stride_t,
        packed_stride_h,
        scale_stride_b,
        scale_stride_block,
        scale_stride_h,
        residual_stride_b,
        residual_stride_t,
        residual_stride_h,
        value_stride_b,
        value_stride_t,
        value_stride_h,
        partial_stride_b,
        partial_stride_h,
        partial_stride_block,
        ml_stride_b,
        ml_stride_h,
        ml_stride_block,
        full_blocks,
        attention_scale,
        NUM_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PACKED_DIM: tl.constexpr,
        USE_RESIDUAL: tl.constexpr,
        RECORD_KEEP: tl.constexpr,
    ):
        block_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        head_index = tl.program_id(2)
        approximate_scores = _q2_block_scores(
            query,
            packed_key,
            scales,
            batch_index,
            head_index,
            block_index,
            query_stride_b,
            query_stride_h,
            packed_stride_b,
            packed_stride_t,
            packed_stride_h,
            scale_stride_b,
            scale_stride_block,
            scale_stride_h,
            attention_scale,
            BLOCK_SIZE=BLOCK_SIZE,
            HEAD_DIM=HEAD_DIM,
            PACKED_DIM=PACKED_DIM,
        )
        block_maximum = tl.max(approximate_scores, axis=0)
        cutoff = tl.load(threshold + batch_index * NUM_HEADS + head_index)
        keep_block = block_maximum >= cutoff
        dim_offsets = tl.arange(0, HEAD_DIM)[None, :]
        output_offsets = tl.arange(0, HEAD_DIM)
        partial_base = (
            batch_index * partial_stride_b
            + head_index * partial_stride_h
            + block_index * partial_stride_block
        )
        ml_base = (
            batch_index * ml_stride_b
            + head_index * ml_stride_h
            + block_index * ml_stride_block
        )
        if keep_block:
            scores = approximate_scores
            token_offsets = tl.arange(0, BLOCK_SIZE)[:, None]
            token_index = block_index * BLOCK_SIZE + token_offsets
            if USE_RESIDUAL:
                query_values = tl.load(
                    query
                    + batch_index * query_stride_b
                    + head_index * query_stride_h
                    + dim_offsets
                ).to(tl.float32)
                residual_values = tl.load(
                    residual
                    + batch_index * residual_stride_b
                    + token_index * residual_stride_t
                    + head_index * residual_stride_h
                    + dim_offsets
                ).to(tl.float32)
                scores += (
                    tl.sum(residual_values * query_values, axis=1) * attention_scale
                )
            maximum = tl.max(scores, axis=0)
            probabilities = tl.exp(scores - maximum)
            normalizer = tl.sum(probabilities, axis=0)
            value_block = tl.load(
                values
                + batch_index * value_stride_b
                + token_index * value_stride_t
                + head_index * value_stride_h
                + dim_offsets
            ).to(tl.float32)
            accumulator = tl.sum(probabilities[:, None] * value_block, axis=0)
            tl.store(partial_output + partial_base + output_offsets, accumulator)
            tl.store(partial_ml + ml_base, maximum)
            tl.store(partial_ml + ml_base + 1, normalizer)
        else:
            tl.store(partial_output + partial_base + output_offsets, 0.0)
            tl.store(partial_ml + ml_base, float("-inf"))
            tl.store(partial_ml + ml_base + 1, 0.0)
        if RECORD_KEEP:
            keep_offset = (
                batch_index * NUM_HEADS + head_index
            ) * full_blocks + block_index
            tl.store(kept + keep_offset, keep_block.to(tl.uint8))

    @triton.jit(do_not_specialize=("tail_len", "tail_part"))
    def _ffd_tail_kernel(
        query,
        tail_key,
        tail_value,
        partial_output,
        partial_ml,
        query_stride_b,
        query_stride_h,
        tail_key_stride_b,
        tail_key_stride_t,
        tail_key_stride_h,
        tail_value_stride_b,
        tail_value_stride_t,
        tail_value_stride_h,
        partial_stride_b,
        partial_stride_h,
        partial_stride_block,
        ml_stride_b,
        ml_stride_h,
        ml_stride_block,
        tail_len,
        tail_part,
        attention_scale,
        NUM_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch_index = program // NUM_HEADS
        head_index = program % NUM_HEADS
        token_offsets = tl.arange(0, BLOCK_SIZE)[:, None]
        dim_offsets = tl.arange(0, HEAD_DIM)[None, :]
        query_values = tl.load(
            query
            + batch_index * query_stride_b
            + head_index * query_stride_h
            + dim_offsets
        ).to(tl.float32)
        key_values = tl.load(
            tail_key
            + batch_index * tail_key_stride_b
            + token_offsets * tail_key_stride_t
            + head_index * tail_key_stride_h
            + dim_offsets,
            mask=token_offsets < tail_len,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key_values * query_values, axis=1) * attention_scale
        scores = tl.where(tl.arange(0, BLOCK_SIZE) < tail_len, scores, float("-inf"))
        maximum = tl.max(scores, axis=0)
        probabilities = tl.exp(scores - maximum)
        probabilities = tl.where(
            tl.arange(0, BLOCK_SIZE) < tail_len, probabilities, 0.0
        )
        normalizer = tl.sum(probabilities, axis=0)
        value_values = tl.load(
            tail_value
            + batch_index * tail_value_stride_b
            + token_offsets * tail_value_stride_t
            + head_index * tail_value_stride_h
            + dim_offsets,
            mask=token_offsets < tail_len,
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.sum(probabilities[:, None] * value_values, axis=0)
        output_offsets = tl.arange(0, HEAD_DIM)
        partial_base = (
            batch_index * partial_stride_b
            + head_index * partial_stride_h
            + tail_part * partial_stride_block
        )
        ml_base = (
            batch_index * ml_stride_b
            + head_index * ml_stride_h
            + tail_part * ml_stride_block
        )
        tl.store(partial_output + partial_base + output_offsets, accumulator)
        tl.store(partial_ml + ml_base, maximum)
        tl.store(partial_ml + ml_base + 1, normalizer)

    @triton.jit
    def _ffd_reduce_kernel(
        partial_output,
        partial_ml,
        output,
        partial_stride_b,
        partial_stride_h,
        partial_stride_block,
        ml_stride_b,
        ml_stride_h,
        ml_stride_block,
        output_stride_b,
        output_stride_h,
        NUM_HEADS: tl.constexpr,
        NUM_PARTS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch_index = program // NUM_HEADS
        head_index = program % NUM_HEADS
        dim_offsets = tl.arange(0, HEAD_DIM)
        global_maximum = float("-inf")
        global_normalizer = 0.0
        global_accumulator = tl.zeros([HEAD_DIM], tl.float32)
        for part in tl.static_range(0, NUM_PARTS):
            ml_base = (
                batch_index * ml_stride_b
                + head_index * ml_stride_h
                + part * ml_stride_block
            )
            part_maximum = tl.load(partial_ml + ml_base)
            part_normalizer = tl.load(partial_ml + ml_base + 1)
            partial_base = (
                batch_index * partial_stride_b
                + head_index * partial_stride_h
                + part * partial_stride_block
            )
            part_accumulator = tl.load(partial_output + partial_base + dim_offsets)
            merged_maximum = tl.maximum(global_maximum, part_maximum)
            global_weight = tl.where(
                global_normalizer > 0, tl.exp(global_maximum - merged_maximum), 0.0
            )
            part_weight = tl.where(
                part_normalizer > 0, tl.exp(part_maximum - merged_maximum), 0.0
            )
            global_accumulator = (
                global_accumulator * global_weight + part_accumulator * part_weight
            )
            global_normalizer = (
                global_normalizer * global_weight + part_normalizer * part_weight
            )
            global_maximum = tl.where(
                part_normalizer > 0, merged_maximum, global_maximum
            )
        result = global_accumulator / global_normalizer
        tl.store(
            output
            + batch_index * output_stride_b
            + head_index * output_stride_h
            + dim_offsets,
            result,
        )


@dataclass
class _Workspace:
    threshold: Any
    partial_output: Any
    partial_ml: Any
    kept: Any
    output: Any


class Gfx1151TritonBackend(_AuditedBackend):
    """Fused block selector-computer specialized for Qwen1.5-MoE MHA."""

    def __init__(self, torch: Any, *, collect_keep_stats: bool = False) -> None:
        super().__init__()
        observed_amd_platform(torch)
        if triton is None:
            raise ContractError("FFD Triton backend requires Triton")
        self.torch = torch
        self.collect_keep_stats = collect_keep_stats
        self.capability = BackendCapability(
            backend_id="triton-q2-sparse-gfx1151-v1",
            available=True,
            platform="hip_gfx1151",
            fused_selector_computer=True,
            performance_claim_allowed=True,
        )
        self._workspaces: dict[tuple[Any, ...], _Workspace] = {}
        self._workspace_lock = threading.Lock()
        self._selected_blocks = 0
        self._total_blocks = 0

    @classmethod
    def build(cls, *, collect_keep_stats: bool = False) -> "Gfx1151TritonBackend":
        try:
            import torch
        except ImportError as exc:
            raise ContractError("FFD Triton backend requires PyTorch") from exc
        return cls(torch, collect_keep_stats=collect_keep_stats)

    def _validate(
        self, query: Any, layer: FFDLayerCache, policy: FFDPolicy
    ) -> tuple[Any, int, int, int]:
        torch = self.torch
        if query.ndim != 4 or query.shape[2] != 1:
            raise ContractError("gfx1151 FFD expects query shape [B, H, 1, D]")
        batch, heads, _, head_dim = query.shape
        if batch != 1 or heads != 16 or head_dim != 128:
            raise ContractError("gfx1151 FFD is specialized to Qwen shape [1,16,1,128]")
        if policy.key_bits != 2:
            raise ContractError("gfx1151 fused FFD currently supports Q2 keys only")
        if policy.block_size not in (64, 128):
            raise ContractError("gfx1151 fused FFD supports block size 64 or 128")
        if policy.graph_mode != "eager":
            raise ContractError(
                "attention-only graph mode was not promoted after the A1 rejection"
            )
        if (
            policy.sink_tokens != policy.block_size
            or policy.local_tokens != policy.block_size
        ):
            raise ContractError(
                "gfx1151 fused FFD v1 requires sink/local windows equal to block_size"
            )
        if not query.is_cuda or query.device != layer.device or not torch.version.hip:
            raise ContractError("gfx1151 FFD tensors must share a HIP device")
        if layer.value.shape[-1] != 128:
            raise ContractError("gfx1151 FFD expects value dimension 128")
        query_bhd = query[:, :, 0, :].contiguous()
        full_blocks = layer.full_token_count // policy.block_size
        tail_len = layer.current_len
        return query_bhd, int(batch), int(full_blocks), int(tail_len)

    def _workspace(
        self,
        query: Any,
        full_blocks: int,
        tail_len: int,
        block_size: int,
    ) -> _Workspace:
        torch = self.torch
        parts = full_blocks + (1 if tail_len else 0)
        key = (
            str(query.device),
            query.dtype,
            parts,
            full_blocks,
            bool(tail_len),
            block_size,
            self.collect_keep_stats,
        )
        with self._workspace_lock:
            value = self._workspaces.get(key)
            if value is None:
                batch, heads, head_dim = query.shape
                value = _Workspace(
                    threshold=torch.empty(
                        (batch, heads), device=query.device, dtype=torch.float32
                    ),
                    partial_output=torch.empty(
                        (batch, heads, parts, head_dim),
                        device=query.device,
                        dtype=torch.float32,
                    ),
                    partial_ml=torch.empty(
                        (batch, heads, parts, 2),
                        device=query.device,
                        dtype=torch.float32,
                    ),
                    kept=torch.empty(
                        (batch, heads, max(full_blocks, 1)),
                        device=query.device,
                        dtype=torch.uint8,
                    ),
                    output=torch.empty_like(query),
                )
                self._workspaces[key] = value
        return value

    def decode(
        self,
        query_states: Any,
        layer_cache: FFDLayerCache,
        policy: FFDPolicy,
        *,
        layer_index: int,
    ) -> Any:
        query, batch, full_blocks, tail_len = self._validate(
            query_states, layer_cache, policy
        )
        parts = full_blocks + (1 if tail_len else 0)
        if parts <= 0:
            raise ContractError("FFD decode cache is empty")
        workspace = self._workspace(query, full_blocks, tail_len, policy.block_size)
        packed_dim = 32
        launch = {"num_warps": 4, "num_stages": 1}
        _ffd_threshold_kernel[(batch * 16,)](
            query,
            layer_cache.k_q,
            layer_cache.k_scale,
            layer_cache.k_tail,
            workspace.threshold,
            full_blocks,
            tail_len,
            query.stride(0),
            query.stride(1),
            layer_cache.k_q.stride(0),
            layer_cache.k_q.stride(1),
            layer_cache.k_q.stride(2),
            layer_cache.k_scale.stride(0),
            layer_cache.k_scale.stride(1),
            layer_cache.k_scale.stride(2),
            layer_cache.k_tail.stride(0),
            layer_cache.k_tail.stride(1),
            layer_cache.k_tail.stride(2),
            attention_scale=128**-0.5,
            delta=policy.delta,
            NUM_HEADS=16,
            BLOCK_SIZE=policy.block_size,
            HEAD_DIM=128,
            PACKED_DIM=packed_dim,
            **launch,
        )
        if full_blocks:
            residual = (
                layer_cache.k_residual
                if layer_cache.k_residual is not None
                else layer_cache.k_q
            )
            _ffd_sparse_block_kernel[(full_blocks, batch, 16)](
                query,
                layer_cache.k_q,
                layer_cache.k_scale,
                residual,
                layer_cache.value,
                workspace.threshold,
                workspace.partial_output,
                workspace.partial_ml,
                workspace.kept,
                query.stride(0),
                query.stride(1),
                layer_cache.k_q.stride(0),
                layer_cache.k_q.stride(1),
                layer_cache.k_q.stride(2),
                layer_cache.k_scale.stride(0),
                layer_cache.k_scale.stride(1),
                layer_cache.k_scale.stride(2),
                residual.stride(0),
                residual.stride(1),
                residual.stride(2),
                layer_cache.value.stride(0),
                layer_cache.value.stride(1),
                layer_cache.value.stride(2),
                workspace.partial_output.stride(0),
                workspace.partial_output.stride(1),
                workspace.partial_output.stride(2),
                workspace.partial_ml.stride(0),
                workspace.partial_ml.stride(1),
                workspace.partial_ml.stride(2),
                full_blocks,
                attention_scale=128**-0.5,
                NUM_HEADS=16,
                BLOCK_SIZE=policy.block_size,
                HEAD_DIM=128,
                PACKED_DIM=packed_dim,
                USE_RESIDUAL=layer_cache.k_residual is not None,
                RECORD_KEEP=self.collect_keep_stats,
                **launch,
            )
        if tail_len:
            _ffd_tail_kernel[(batch * 16,)](
                query,
                layer_cache.k_tail,
                layer_cache.v_tail,
                workspace.partial_output,
                workspace.partial_ml,
                query.stride(0),
                query.stride(1),
                layer_cache.k_tail.stride(0),
                layer_cache.k_tail.stride(1),
                layer_cache.k_tail.stride(2),
                layer_cache.v_tail.stride(0),
                layer_cache.v_tail.stride(1),
                layer_cache.v_tail.stride(2),
                workspace.partial_output.stride(0),
                workspace.partial_output.stride(1),
                workspace.partial_output.stride(2),
                workspace.partial_ml.stride(0),
                workspace.partial_ml.stride(1),
                workspace.partial_ml.stride(2),
                tail_len,
                full_blocks,
                attention_scale=128**-0.5,
                NUM_HEADS=16,
                BLOCK_SIZE=policy.block_size,
                HEAD_DIM=128,
                **launch,
            )
        _ffd_reduce_kernel[(batch * 16,)](
            workspace.partial_output,
            workspace.partial_ml,
            workspace.output,
            workspace.partial_output.stride(0),
            workspace.partial_output.stride(1),
            workspace.partial_output.stride(2),
            workspace.partial_ml.stride(0),
            workspace.partial_ml.stride(1),
            workspace.partial_ml.stride(2),
            workspace.output.stride(0),
            workspace.output.stride(1),
            NUM_HEADS=16,
            NUM_PARTS=parts,
            HEAD_DIM=128,
            **launch,
        )
        if self.collect_keep_stats and full_blocks:
            selected = int(workspace.kept[:, :, :full_blocks].sum().item())
            self._selected_blocks += selected
            self._total_blocks += batch * 16 * full_blocks
        self._record(layer_index)
        return workspace.output

    def audit(self) -> dict[str, Any]:
        value = super().audit()
        value["selector"] = {
            "statistics_enabled": self.collect_keep_stats,
            "selected_blocks": self._selected_blocks,
            "total_blocks": self._total_blocks,
            "keep_ratio": (
                self._selected_blocks / self._total_blocks
                if self._total_blocks
                else None
            ),
        }
        return value


__all__ = ["Gfx1151TritonBackend"]
