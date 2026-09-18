"""Create draft model manifests from pinned Hugging Face repository metadata.

The importer intentionally downloads only small control files.  Safetensors
payload identity comes from Hugging Face LFS metadata and must later be checked
against locally downloaded files before a manifest can become ``frozen``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


Fetcher = Callable[[str], bytes]

_HF_ORIGIN = "https://huggingface.co"
_MAX_SMALL_FILE_BYTES = 64 * 1024 * 1024
_TOKENIZER_FILE_ORDER = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer.model.v1",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)
_SAFETENSORS_DTYPE_NAMES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "U32": "uint32",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "U64": "uint64",
    "F8_E4M3": "float8_e4m3",
    "F8_E5M2": "float8_e5m2",
}


class HuggingFaceImportError(ValueError):
    """Raised when pinned Hugging Face metadata cannot form a safe manifest."""


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_revision(revision: str) -> None:
    if not _is_lower_hex(revision, 40):
        raise HuggingFaceImportError(
            "revision must be an immutable 40-character lowercase hexadecimal commit; "
            "floating revisions such as 'main', tags, and branch names are forbidden"
        )


def _validate_repo_id(repo_id: str) -> None:
    segments = repo_id.split("/")
    if (
        not repo_id
        or repo_id.startswith("/")
        or repo_id.endswith("/")
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise HuggingFaceImportError(f"invalid Hugging Face repository id: {repo_id!r}")


def _metadata_endpoint(repo_id: str, revision: str) -> str:
    encoded_repo = quote(repo_id, safe="/")
    return f"{_HF_ORIGIN}/api/models/{encoded_repo}/revision/{revision}?blobs=true"


def _resolve_url(repo_id: str, revision: str, path: str) -> str:
    encoded_repo = quote(repo_id, safe="/")
    encoded_path = quote(path, safe="/")
    return f"{_HF_ORIGIN}/{encoded_repo}/resolve/{revision}/{encoded_path}"


def _default_fetcher(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json, application/octet-stream",
            "User-Agent": "uma-qmoe-huggingface-importer/1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS origin
            return response.read(_MAX_SMALL_FILE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise HuggingFaceImportError(f"cannot fetch {url}: {exc}") from exc


def _fetch_bytes(fetcher: Fetcher, url: str, description: str) -> bytes:
    try:
        payload = fetcher(url)
    except HuggingFaceImportError:
        raise
    except Exception as exc:
        raise HuggingFaceImportError(f"cannot fetch {description} from {url}: {exc}") from exc
    if not isinstance(payload, bytes):
        raise HuggingFaceImportError(
            f"fetcher must return bytes for {description}; received {type(payload).__name__}"
        )
    if not payload:
        raise HuggingFaceImportError(f"{description} is empty")
    if len(payload) > _MAX_SMALL_FILE_BYTES:
        raise HuggingFaceImportError(
            f"{description} exceeds the {_MAX_SMALL_FILE_BYTES}-byte control-file limit"
        )
    return payload


def _decode_json(payload: bytes, description: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceImportError(f"{description} is not valid UTF-8 JSON: {exc}") from exc


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HuggingFaceImportError(f"{description} must be a JSON object")
    return value


def _safe_repository_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise HuggingFaceImportError("Hugging Face sibling is missing a non-empty rfilename")
    segments = value.split("/")
    if value.startswith("/") or any(segment in {"", ".", ".."} for segment in segments):
        raise HuggingFaceImportError(f"unsafe repository path in metadata: {value!r}")
    return value


def _sibling_map(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise HuggingFaceImportError("Hugging Face API metadata must contain a siblings array")

    by_path: dict[str, Mapping[str, Any]] = {}
    for position, raw_sibling in enumerate(siblings):
        sibling = _require_mapping(raw_sibling, f"siblings[{position}]")
        path = _safe_repository_path(sibling.get("rfilename"))
        if path in by_path:
            raise HuggingFaceImportError(f"duplicate sibling metadata for {path!r}")
        by_path[path] = sibling
    return by_path


def _file_identity(path: str, payload: bytes, sibling: Mapping[str, Any]) -> dict[str, Any]:
    declared_size = sibling.get("size")
    if declared_size is not None:
        if isinstance(declared_size, bool) or not isinstance(declared_size, int):
            raise HuggingFaceImportError(f"invalid API size for {path!r}")
        if declared_size != len(payload):
            raise HuggingFaceImportError(
                f"size mismatch for {path!r}: API declares {declared_size}, fetched {len(payload)}"
            )
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fetch_control_file(
    fetcher: Fetcher,
    repo_id: str,
    revision: str,
    path: str,
    siblings: Mapping[str, Mapping[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    if path.endswith(".safetensors"):
        raise HuggingFaceImportError(
            f"refusing to download weight payload {path!r}; use LFS metadata instead"
        )
    try:
        sibling = siblings[path]
    except KeyError as exc:
        raise HuggingFaceImportError(f"required repository file {path!r} is missing") from exc
    payload = _fetch_bytes(fetcher, _resolve_url(repo_id, revision, path), path)
    return payload, _file_identity(path, payload, sibling)


def _require_integer(config: Mapping[str, Any], field: str, *aliases: str) -> int:
    for name in (field, *aliases):
        value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    rendered = ", ".join(repr(name) for name in (field, *aliases))
    raise HuggingFaceImportError(f"config must declare a positive integer in one of: {rendered}")


def _config_dtype(config: Mapping[str, Any]) -> str:
    declarations = [
        config[name]
        for name in ("torch_dtype", "dtype")
        if isinstance(config.get(name), str) and config[name]
    ]
    if not declarations:
        raise HuggingFaceImportError("config must declare torch_dtype or dtype")
    if len(set(declarations)) != 1:
        raise HuggingFaceImportError("config torch_dtype and dtype declarations disagree")
    return declarations[0]


def _parse_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise HuggingFaceImportError("config.architectures must contain a model class name")

    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise HuggingFaceImportError("config.model_type must be a non-empty string")

    normalize_value: object = None
    for name in ("normalize_top_k_probability", "norm_topk_prob"):
        if name in config:
            normalize_value = config[name]
            break
    if not isinstance(normalize_value, bool):
        raise HuggingFaceImportError(
            "config must declare boolean normalize_top_k_probability or norm_topk_prob"
        )

    return {
        "class_name": architectures[0],
        "model_type": model_type,
        "num_layers": _require_integer(config, "num_hidden_layers", "num_layers"),
        "hidden_size": _require_integer(config, "hidden_size"),
        "expert_intermediate_size": _require_integer(
            config, "intermediate_size", "moe_intermediate_size", "expert_intermediate_size"
        ),
        "num_experts": _require_integer(config, "num_experts", "num_local_experts"),
        "top_k": _require_integer(
            config, "num_experts_per_tok", "num_selected_experts", "top_k"
        ),
        "max_position_embeddings": _require_integer(config, "max_position_embeddings"),
        "normalize_top_k_probability": normalize_value,
    }


def _lfs_weight_identity(path: str, sibling: Mapping[str, Any]) -> dict[str, Any]:
    lfs = sibling.get("lfs")
    if not isinstance(lfs, Mapping):
        raise HuggingFaceImportError(
            f"weight {path!r} has no LFS metadata; refusing an unverifiable draft"
        )
    sha256 = lfs.get("sha256")
    if not _is_lower_hex(sha256, 64):
        raise HuggingFaceImportError(f"weight {path!r} is missing a valid LFS SHA-256")
    size = lfs.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HuggingFaceImportError(f"weight {path!r} is missing a valid positive LFS size")
    return {"path": path, "size_bytes": size, "sha256": sha256}


def _uploaded_dtype_from_api(
    metadata: Mapping[str, Any], model_card_claim: str
) -> tuple[str, dict[str, int], bool]:
    safetensors = metadata.get("safetensors")
    parameters = safetensors.get("parameters") if isinstance(safetensors, Mapping) else None
    if parameters is None:
        raise HuggingFaceImportError(
            "Hugging Face API metadata is missing safetensors parameter dtype/counts; "
            "refusing to create a draft with unknown uploaded-weight identity"
        )
    if not isinstance(parameters, Mapping) or not parameters:
        raise HuggingFaceImportError(
            "Hugging Face API safetensors.parameters must be a non-empty object when present"
        )

    declarations: list[tuple[str, str, int]] = []
    for raw_dtype, count in parameters.items():
        if not isinstance(raw_dtype, str) or not raw_dtype:
            raise HuggingFaceImportError(
                "Hugging Face API safetensors.parameters contains an invalid dtype"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise HuggingFaceImportError(
                f"Hugging Face API parameter count for {raw_dtype!r} must be positive"
            )
        normalized = _SAFETENSORS_DTYPE_NAMES.get(raw_dtype, raw_dtype.lower())
        declarations.append((raw_dtype, normalized, count))
    declarations.sort(key=lambda item: item[0])

    normalized_dtypes = sorted({normalized for _, normalized, _ in declarations})
    uploaded_dtype = (
        normalized_dtypes[0]
        if len(normalized_dtypes) == 1
        else "mixed[" + ",".join(normalized_dtypes) + "]"
    )
    api_counts = {raw_dtype: count for raw_dtype, _, count in declarations}
    normalized_claim = _SAFETENSORS_DTYPE_NAMES.get(
        model_card_claim.upper(), model_card_claim.lower()
    )
    return uploaded_dtype, api_counts, normalized_claim != uploaded_dtype


def _weight_paths_from_index(index: Mapping[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise HuggingFaceImportError("safetensors index must contain a non-empty weight_map")
    paths: set[str] = set()
    for tensor_name, raw_path in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise HuggingFaceImportError("safetensors index contains an invalid tensor name")
        path = _safe_repository_path(raw_path)
        if not path.endswith(".safetensors"):
            raise HuggingFaceImportError(
                f"safetensors index maps {tensor_name!r} to non-safetensors file {path!r}"
            )
        paths.add(path)
    return sorted(paths)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_model_manifest_draft(
    repo_id: str,
    revision: str,
    *,
    model_card_uploaded_dtype: str,
    oracle_dtype: str,
    variant: str = "base",
    fetcher: Fetcher | None = None,
    api_metadata: Mapping[str, Any] | None = None,
    resolved_at: str | None = None,
    declared_total_parameters: str | None = None,
    declared_active_parameters: str | None = None,
) -> dict[str, Any]:
    """Build a schema-compatible draft manifest without downloading weights.

    ``model_card_uploaded_dtype`` is deliberately caller supplied as a claim,
    not as the observed upload dtype.  The latter comes from the API's
    ``safetensors.parameters`` inventory.  Config dtype, API inventory, model-
    card claim, and oracle compute dtype are distinct facts and must not be
    inferred from one another.
    """

    _validate_repo_id(repo_id)
    _validate_revision(revision)
    if variant not in {"base", "sft", "instruct"}:
        raise HuggingFaceImportError("variant must be one of: base, sft, instruct")
    if not isinstance(model_card_uploaded_dtype, str) or not model_card_uploaded_dtype:
        raise HuggingFaceImportError("model_card_uploaded_dtype must be a non-empty string")
    if not isinstance(oracle_dtype, str) or not oracle_dtype:
        raise HuggingFaceImportError("oracle_dtype must be a non-empty string")
    for name, value in (
        ("declared_total_parameters", declared_total_parameters),
        ("declared_active_parameters", declared_active_parameters),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise HuggingFaceImportError(f"{name} must be a non-empty string when supplied")

    actual_fetcher = fetcher or _default_fetcher
    endpoint = _metadata_endpoint(repo_id, revision)
    if api_metadata is None:
        raw_metadata = _fetch_bytes(actual_fetcher, endpoint, "Hugging Face API metadata")
        metadata = _require_mapping(
            _decode_json(raw_metadata, "Hugging Face API metadata"),
            "Hugging Face API metadata",
        )
    else:
        metadata = _require_mapping(api_metadata, "Hugging Face API metadata")

    metadata_sha = metadata.get("sha")
    if metadata_sha != revision:
        raise HuggingFaceImportError(
            f"Hugging Face API resolved revision to {metadata_sha!r}, expected {revision!r}"
        )
    metadata_id = metadata.get("id")
    if metadata_id is not None and metadata_id != repo_id:
        raise HuggingFaceImportError(
            f"Hugging Face API returned repository {metadata_id!r}, expected {repo_id!r}"
        )
    uploaded_dtype, api_parameter_counts, evidence_conflict = _uploaded_dtype_from_api(
        metadata, model_card_uploaded_dtype
    )

    siblings = _sibling_map(metadata)
    config_payload, config_identity = _fetch_control_file(
        actual_fetcher, repo_id, revision, "config.json", siblings
    )
    config = _require_mapping(_decode_json(config_payload, "config.json"), "config.json")
    architecture = _parse_architecture(config)
    if declared_total_parameters is not None:
        architecture["declared_total_parameters"] = declared_total_parameters
    if declared_active_parameters is not None:
        architecture["declared_active_parameters"] = declared_active_parameters

    tokenizer_files: list[dict[str, Any]] = []
    tokenizer_config: Mapping[str, Any] = {}
    for path in _TOKENIZER_FILE_ORDER:
        if path not in siblings:
            continue
        payload, identity = _fetch_control_file(
            actual_fetcher, repo_id, revision, path, siblings
        )
        tokenizer_files.append(identity)
        if path == "tokenizer_config.json":
            tokenizer_config = _require_mapping(
                _decode_json(payload, "tokenizer_config.json"), "tokenizer_config.json"
            )
    if "tokenizer_config.json" not in siblings:
        raise HuggingFaceImportError("required repository file 'tokenizer_config.json' is missing")

    chat_template = tokenizer_config.get("chat_template")
    if chat_template is not None and not isinstance(chat_template, str):
        raise HuggingFaceImportError("tokenizer_config.json chat_template must be a string or null")

    index_path = "model.safetensors.index.json"
    index_identity: dict[str, Any] | None = None
    if index_path in siblings:
        index_payload, index_identity = _fetch_control_file(
            actual_fetcher, repo_id, revision, index_path, siblings
        )
        index = _require_mapping(
            _decode_json(index_payload, index_path), index_path
        )
        weight_paths = _weight_paths_from_index(index)
    else:
        weight_paths = sorted(path for path in siblings if path.endswith(".safetensors"))
        if not weight_paths:
            raise HuggingFaceImportError(
                "repository contains neither a safetensors index nor a safetensors weight"
            )

    weight_artifacts: list[dict[str, Any]] = []
    for path in weight_paths:
        try:
            sibling = siblings[path]
        except KeyError as exc:
            raise HuggingFaceImportError(
                f"safetensors index references missing repository file {path!r}"
            ) from exc
        weight_artifacts.append(_lfs_weight_identity(path, sibling))
    if index_identity is not None:
        weight_artifacts.append(index_identity)

    card_data = metadata.get("cardData")
    license_name: object = None
    if isinstance(card_data, Mapping):
        license_name = card_data.get("license")
    if license_name is None:
        license_name = metadata.get("license")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "model_manifest",
        "status": "draft",
        "model_id": repo_id,
        "model_revision": revision,
        "variant": variant,
        "config": config_identity,
        "architecture": architecture,
        "tokenizer": {
            "repository": repo_id,
            "revision": revision,
            "chat_template": chat_template,
            "files": tokenizer_files,
        },
        "dtypes": {
            "config_declared": _config_dtype(config),
            "uploaded_weights": uploaded_dtype,
            "uploaded_weights_evidence": (
                "Hugging Face API safetensors.parameters at pinned revision"
            ),
            "api_reported_parameter_counts": api_parameter_counts,
            "model_card_uploaded_weights_claim": model_card_uploaded_dtype,
            "evidence_conflict": evidence_conflict,
            "oracle_compute": oracle_dtype,
            "local_tensor_scan": "pending_download",
        },
        "weights": {
            "format": "safetensors",
            "hash_source": "huggingface_lfs_metadata",
            "local_verification": "pending_download",
            "tensor_hashes_status": "pending_local_scan",
            "artifacts": weight_artifacts,
        },
        "provenance": {
            "provider": "huggingface",
            "repository": repo_id,
            "resolved_at": resolved_at or _utc_now(),
            "metadata_endpoint": endpoint,
        },
    }
    if isinstance(license_name, str) and license_name:
        manifest["license"] = license_name
    return manifest


__all__ = ["Fetcher", "HuggingFaceImportError", "build_model_manifest_draft"]
