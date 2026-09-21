"""Small, resumable control plane for pre-contract quantization experiments.

Everything emitted here is exploratory L0 evidence.  Promotion into the existing
contract/TargetPack path is deliberately a separate, explicit operation.
"""

from .ledger import JsonlLedger, trial_identity
from .quantizers import QuantizedCandidate, Quantizer, RTNQuantizer
from .types import (
    CalibrationView,
    Encoding,
    EvidenceLevel,
    TrialResult,
    TrialSpec,
    TrialStatus,
    Unit,
)

__all__ = [
    "CalibrationView",
    "Encoding",
    "EvidenceLevel",
    "JsonlLedger",
    "QuantizedCandidate",
    "Quantizer",
    "RTNQuantizer",
    "TrialResult",
    "TrialSpec",
    "TrialStatus",
    "Unit",
    "trial_identity",
]
