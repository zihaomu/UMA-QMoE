"""Faster Flash Decoding research backend for Qwen1.5-MoE."""

from .backend import (
    BackendCapability,
    TorchOracleBackend,
    UnavailableBackend,
    backend_source_sha256,
    observed_amd_platform,
)
from .cache import DenseLayerCache, FFDCache, FFDLayerCache
from .integration import QwenFFDInstallation, install_qwen_ffd
from .oracle import (
    dense_attention,
    exact_attention_scores,
    q2_sparse_attention,
    selector_fidelity_report,
)
from .policy import FFDPolicy
from .quantization import (
    QuantizedKeyBlocks,
    dequantize_key_blocks,
    pack_quantized,
    quantize_key_blocks,
    unpack_quantized,
)
from .triton_backend import Gfx1151TritonBackend

__all__ = [
    "BackendCapability",
    "DenseLayerCache",
    "FFDCache",
    "FFDLayerCache",
    "FFDPolicy",
    "Gfx1151TritonBackend",
    "QuantizedKeyBlocks",
    "QwenFFDInstallation",
    "TorchOracleBackend",
    "UnavailableBackend",
    "backend_source_sha256",
    "dequantize_key_blocks",
    "dense_attention",
    "exact_attention_scores",
    "install_qwen_ffd",
    "observed_amd_platform",
    "pack_quantized",
    "q2_sparse_attention",
    "quantize_key_blocks",
    "selector_fidelity_report",
    "unpack_quantized",
]
