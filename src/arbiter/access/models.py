"""Access auditor models.

Pydantic v2 models matching the access_auditor contract interface.
All models use frozen config for immutability. IntEnum-like ordering
on DataTier and FindingSeverity via integer comparison.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DataTier",
    "FindingCode",
    "FindingSeverity",
    "SlotDecisionVerdict",
    "NodeId",
    "FieldPath",
    "AdapterSlotId",
    "FieldEntry",
    "SchemaWarning",
    "WalkResult",
    "ClassifiedField",
    "ClassificationResult",
    "StructuralProfile",
    "AccessFinding",
    "AccessFindingEvidence",
    "GateConfig",
    "SlotDecision",
    "DeclaredAccess",
    "ClassificationRegistryEntry",
    "ObservedOutput",
]


# --- Enums ---


class DataTier(IntEnum):
    """Data classification tier. Integer value determines severity ordering.

    PUBLIC(0) < INTERNAL(1) < CONFIDENTIAL(2) < RESTRICTED(3).
    """

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


class FindingCode(StrEnum):
    """Stable string identifiers used for access gating and ledger records."""

    C005 = "C005"
    FA_A_015 = "FA_A_015"
    INCOMPLETE_SCHEMA = "INCOMPLETE_SCHEMA"


class FindingSeverity(IntEnum):
    """Finding severity levels. Integer value determines ordering.

    INFO(0) < WARNING(1) < ERROR(2) < CRITICAL(3).
    """

    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class SlotDecisionVerdict(str):
    """Verdict for a slot gating decision."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"

    def __new__(cls, value: str) -> SlotDecisionVerdict:
        obj = str.__new__(cls, value)
        return obj


# --- NewType primitives ---

NodeId = NewType("NodeId", str)
FieldPath = NewType("FieldPath", str)
AdapterSlotId = NewType("AdapterSlotId", str)


# --- Pydantic models ---


class FieldEntry(BaseModel):
    """A single leaf field discovered by the schema walker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    field_type: str
    nullable: bool
    format_hint: str = ""


class SchemaWarning(BaseModel):
    """Warning generated during schema walking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str  # Always FindingCode.INCOMPLETE_SCHEMA for walker warnings
    path: str
    message: str


class WalkResult(BaseModel):
    """Result of walking an OpenAPI response schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: list[FieldEntry]
    warnings: list[SchemaWarning]


class ClassifiedField(BaseModel):
    """A field entry paired with its resolved data classification tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    tier: DataTier
    matched_pattern: str = ""


class ClassificationResult(BaseModel):
    """Result of classifying a list of field entries against the registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classified_fields: list[ClassifiedField]
    tier_set: list[DataTier]


class StructuralProfile(BaseModel):
    """Structural access profile for a node at a given endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    adapter_slot_id: str
    endpoint: str
    tiers: list[DataTier]
    classified_fields: list[ClassifiedField]
    warnings: list[SchemaWarning]
    schema_complete: bool
    computed_at: str


class AccessFindingEvidence(BaseModel):
    """Structured evidence attached to an AccessFinding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    structural_tiers: list[DataTier]
    declared_tiers: list[DataTier]
    undeclared_tiers: list[DataTier]
    fields_by_undeclared_tier: dict[str, list[str]]
    endpoint: str


class AccessFinding(BaseModel):
    """A finding produced by the access auditor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: FindingSeverity
    node_id: str
    adapter_slot_id: str
    message: str = Field(min_length=10)
    evidence: AccessFindingEvidence
    timestamp: str


class GateConfig(BaseModel):
    """Configuration for slot gating decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_on_codes: list[FindingCode]
    assume_worst_on_incomplete: bool = True


class SlotDecision(BaseModel):
    """Gating decision for an adapter slot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: str  # SlotDecisionVerdict value
    adapter_slot_id: str
    node_id: str
    findings: list[AccessFinding]
    blocking_codes: list[str]
    profile: StructuralProfile
    decided_at: str


class DeclaredAccess(BaseModel):
    """Declared data access for a node from the access graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    declared_read_tiers: list[DataTier]
    declared_write_tiers: list[DataTier] = []


class ClassificationRegistryEntry(BaseModel):
    """A single entry in the classification registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_pattern: str = Field(min_length=1)
    tier: DataTier
    pattern_type: str  # 'fnmatch' or 'regex'
    description: str | None = None


class ObservedOutput(BaseModel):
    """Runtime observation of actual data tiers present in a node's output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    adapter_slot_id: str
    observed_tiers: list[DataTier]
    observed_fields: list[ClassifiedField]
    observed_at: str
