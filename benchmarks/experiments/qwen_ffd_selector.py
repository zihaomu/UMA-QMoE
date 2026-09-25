#!/usr/bin/env python3
"""Capture per-layer/head Qwen FFD selector fidelity without cache dumps."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from uma_qmoe.ffd import selector_fidelity_report
from uma_qmoe.native_backend import install_mixed_target_backend
from uma_qmoe.qwen_compressed_loader import load_fixed_qwen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--target-policy-id", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--native-build-dir", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--delta", type=float, choices=(4, 5, 6, 7, 8), default=7)
    parser.add_argument("--key-bits", type=int, choices=(2, 4), default=2)
    parser.add_argument("--block-size", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--sink-tokens", type=int)
    parser.add_argument("--local-tokens", type=int)
    return parser


def _fixture_text(path: Path) -> str:
    texts = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"fixture row {line_number} is not an object")
        text = value.get("prompt") or value.get("text")
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"fixture row {line_number} has no prompt/text")
        texts.append(text)
    if not texts:
        raise RuntimeError("prompt fixture is empty")
    return "\n\n".join(texts)


def _fixed_length_ids(tokenizer: Any, text: str, token_count: int, torch: Any) -> Any:
    if token_count <= 0 or token_count > 8192:
        raise RuntimeError("context-tokens must be in [1, 8192]")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError("fixture tokenized to zero tokens")
    repeats = (token_count + len(ids) - 1) // len(ids)
    return torch.tensor([((ids * repeats)[:token_count])], dtype=torch.long)


def main() -> int:
    args = _parser().parse_args()
    sink_tokens = args.sink_tokens or args.block_size
    local_tokens = args.local_tokens or args.block_size
    if args.context_tokens < args.block_size:
        raise RuntimeError("selector capture needs at least one complete block")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.qwen2_moe.modeling_qwen2_moe import apply_rotary_pos_emb

    if not torch.version.hip:
        raise RuntimeError("Qwen FFD selector capture requires PyTorch HIP")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    input_ids = _fixed_length_ids(
        tokenizer, _fixture_text(args.prompt_fixture), args.context_tokens, torch
    ).to("cuda:0")
    native = install_mixed_target_backend(build_directory=args.native_build_dir)
    rows: list[dict[str, Any]] = []
    handles = []

    def capture(layer_index: int):
        def hook(
            module: Any, positional: tuple[Any, ...], keyword: dict[str, Any]
        ) -> None:
            hidden_states = keyword.get("hidden_states")
            if hidden_states is None:
                hidden_states = positional[0]
            position_embeddings = keyword.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(
                    "Qwen selector hook did not receive position embeddings"
                )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(
                query, key, position_embeddings[0], position_embeddings[1]
            )
            report = selector_fidelity_report(
                query[:, :, -1, :],
                key.transpose(1, 2),
                value.transpose(1, 2),
                delta=args.delta,
                block_size=args.block_size,
                sink_tokens=sink_tokens,
                local_tokens=local_tokens,
                key_bits=args.key_bits,
            )
            rows.append(
                {
                    "schema_version": 1,
                    "kind": "ffd_selector_fidelity",
                    "evidence_level": "L0",
                    "layer_index": layer_index,
                    "report": report,
                }
            )

        return hook

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with load_fixed_qwen(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=True,
        target_policy_id=args.target_policy_id,
    ) as host:
        for layer_index, layer in enumerate(host.model.model.layers):
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    capture(layer_index), with_kwargs=True
                )
            )
        try:
            with torch.inference_mode():
                host.model(input_ids=input_ids, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        loader_evidence = dict(host.evidence)

    if len(rows) != 24:
        raise RuntimeError(f"expected 24 selector rows, captured {len(rows)}")
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    recalls = []
    false_negatives = []
    selected_mass = []
    for row in rows:
        per_head = row["report"]["per_head"]
        recalls.extend(per_head["block_recall"][0])
        false_negatives.extend(per_head["false_negative_rate"][0])
        selected_mass.extend(per_head["selected_mass"][0])
    decision = {
        "schema_version": 1,
        "kind": "ffd_selector_decision",
        "status": (
            "passed"
            if sum(recalls) / len(recalls) >= 0.999
            and sum(false_negatives) / len(false_negatives) <= 0.001
            and sum(selected_mass) / len(selected_mass) >= 0.995
            and sorted(selected_mass)[max(0, math.ceil(0.01 * len(selected_mass)) - 1)]
            >= 0.98
            else "rejected"
        ),
        "gates": {
            "mean_block_recall": sum(recalls) / len(recalls),
            "mean_salient_token_false_negative_rate": (
                sum(false_negatives) / len(false_negatives)
            ),
            "mean_selected_attention_mass": sum(selected_mass) / len(selected_mass),
            "p01_selected_attention_mass": sorted(selected_mass)[
                max(0, math.ceil(0.01 * len(selected_mass)) - 1)
            ],
        },
        "context_tokens": args.context_tokens,
        "delta": args.delta,
        "key_bits": args.key_bits,
        "block_size": args.block_size,
        "target_policy_id": args.target_policy_id,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "native_platform": native.platform,
        "expert_pack_mapping_count": loader_evidence["expert_pack_mapping_count"],
        "selector_jsonl": str(args.output),
    }
    decision_path = args.output.with_suffix(".decision.json")
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
