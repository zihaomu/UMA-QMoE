"""Offline full-model BF16 smoke evidence for the frozen OLMoE Oracle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping

from .contracts import ContractError, canonical_sha256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _select_fixture_prompt(path: Path, prompt_id: str) -> str:
    if not prompt_id.strip():
        raise ContractError("prompt_id must be non-empty")
    matches: list[str] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read prompt fixture {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid JSON in prompt fixture at line {line_number}: {exc}"
            ) from exc
        if not isinstance(item, dict):
            raise ContractError(f"prompt fixture line {line_number} must be an object")
        item_id = item.get("id")
        prompt = item.get("prompt")
        if not isinstance(item_id, str) or not item_id:
            raise ContractError(f"prompt fixture line {line_number} has no valid id")
        if item_id in seen_ids:
            raise ContractError(f"duplicate prompt fixture id {item_id!r}")
        seen_ids.add(item_id)
        if item_id == prompt_id:
            if not isinstance(prompt, str) or not prompt:
                raise ContractError(f"prompt {prompt_id!r} is empty or not a string")
            matches.append(prompt)
    if len(matches) != 1:
        raise ContractError(f"prompt fixture must contain exactly one id {prompt_id!r}")
    return matches[0]


def run_oracle_smoke(
    derivation: Mapping[str, Any],
    model_directory: str | Path,
    prompt_fixture: str | Path,
    *,
    target_id: str,
    prompt_id: str,
    derivation_path: str,
    derivation_file_sha256: str,
    prompt_fixture_path: str,
    prompt_fixture_file_sha256: str,
) -> dict[str, Any]:
    """Load the complete local BF16 model and run one deterministic forward pass."""

    if derivation.get("kind") != "model_derivation":
        raise ContractError("oracle smoke requires a ModelDerivation")
    if derivation.get("status") != "verified":
        raise ContractError("oracle smoke requires a verified ModelDerivation")
    if derivation["transform"]["target_dtype"] != "BF16":
        raise ContractError("oracle smoke requires a BF16 derivation")
    if not target_id.strip():
        raise ContractError("target_id must be non-empty")

    model_path = Path(model_directory)
    if not model_path.is_dir():
        raise ContractError(f"model directory does not exist: {model_path}")
    prompt = _select_fixture_prompt(Path(prompt_fixture), prompt_id)

    # These switches are set before importing Transformers. The smoke must be
    # satisfiable exclusively by the locally verified artifact directory.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ContractError("oracle smoke requires torch and transformers") from exc

    if not torch.cuda.is_available():
        raise ContractError("no CUDA/HIP device is available to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise ContractError("the selected accelerator does not report BF16 support")

    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    observed = {
        "model_type": getattr(config, "model_type", None),
        "num_experts": getattr(config, "num_experts", None),
        "num_experts_per_token": getattr(config, "num_experts_per_tok", None),
    }
    expected = {
        "model_type": "olmoe",
        "num_experts": 64,
        "num_experts_per_token": 8,
    }
    if observed != expected:
        raise ContractError(
            f"unexpected OLMoE runtime architecture: {observed!r}, expected {expected!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0].tolist()
    if not token_ids:
        raise ContractError("fixed prompt encoded to zero tokens")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    device = torch.device("cuda:0")
    inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    forward_started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
        final_logits = output.logits[:, -1, :].float()
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_started

    finite = bool(torch.isfinite(final_logits).all().item())
    if not finite:
        raise ContractError("Oracle smoke produced NaN or Inf logits")
    top_values, top_indices = torch.topk(final_logits, k=5, dim=-1)
    logits_cpu = final_logits.detach().cpu().contiguous()
    logits_sha256 = hashlib.sha256(logits_cpu.numpy().tobytes()).hexdigest()
    capability = torch.cuda.get_device_capability(0)

    return {
        "schema_version": 1,
        "kind": "oracle_smoke",
        "generated_at": _utc_now(),
        "target_id": target_id.strip(),
        "status": "passed",
        "model": {
            "model_id": derivation["source"]["model_id"],
            "model_revision": derivation["source"]["model_revision"],
            "derivation_id": derivation["derivation_id"],
            "derivation_path": derivation_path,
            "derivation_file_sha256": derivation_file_sha256,
            "derivation_semantic_sha256": canonical_sha256(derivation),
            "artifact_root": derivation["artifact_root"],
            "weight_dtype": "BF16",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "accelerator_name": torch.cuda.get_device_name(0),
            "compute_capability": (
                list(capability) if torch.version.cuda is not None else None
            ),
            "bf16_supported": True,
            **observed,
        },
        "input": {
            "fixture_path": prompt_fixture_path,
            "fixture_file_sha256": prompt_fixture_file_sha256,
            "prompt_id": prompt_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids": token_ids,
        },
        "timing": {
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
        },
        "memory": {
            "free_before_load_bytes": free_before,
            "total_bytes": total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "result": {
            "finite": True,
            "logits_shape": list(final_logits.shape),
            "final_token_logits_sha256": logits_sha256,
            "top_token_ids": top_indices[0].detach().cpu().tolist(),
            "top_logits": top_values[0].detach().cpu().tolist(),
        },
    }


__all__ = ["_select_fixture_prompt", "run_oracle_smoke"]
