"""Factory and utility functions for Arbiter models. No business logic -- only
construction, parsing, serialization, and classification helpers."""

from __future__ import annotations

import fnmatch
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Union

from pydantic import ValidationError

from .enums import DataTier, TrustEventType, TrustTier
from .graph import AccessGraph, AccessGraphNode
from .signals import ClassificationRule
from .trust import LedgerCheckpoint, LedgerLine, TrustLedgerEntry
from .types import NodeId, SequenceNumber, Sha256Hex, TrustScore

if TYPE_CHECKING:
    from .api import ErrorResponse

__all__ = [
    "create_trust_ledger_entry",
    "create_ledger_checkpoint",
    "build_access_graph",
    "parse_ledger_line",
    "serialize_ledger_line",
    "create_error_response",
    "validate_canary_fingerprint",
    "score_to_tier",
    "classify_field",
]

_UUID_V4_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


def _utc_now() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_trust_ledger_entry(
    node: NodeId,
    event: TrustEventType,
    weight: float,
    score_before: TrustScore,
    sequence_number: SequenceNumber,
    detail: str,
) -> TrustLedgerEntry:
    """Create a validated TrustLedgerEntry with automatic timestamp.

    score_after = clamp(score_before + weight, 0.0, 1.0).
    """
    if event in (TrustEventType.AUDIT_FAIL, TrustEventType.ACCESS_VIOLATION) and not detail:
        raise ValueError("detail must be non-empty for AUDIT_FAIL and ACCESS_VIOLATION events")

    score_after = max(0.0, min(1.0, score_before + weight))

    return TrustLedgerEntry(
        ts=_utc_now(),
        node=node,
        event=event,
        weight=weight,
        score_before=score_before,
        score_after=score_after,
        sequence_number=sequence_number,
        detail=detail,
    )


def create_ledger_checkpoint(
    sequence_number: SequenceNumber,
    checksum: Sha256Hex,
    entry_count: int,
) -> LedgerCheckpoint:
    """Create a validated LedgerCheckpoint with automatic timestamp."""
    return LedgerCheckpoint(
        ts=_utc_now(),
        sequence_number=sequence_number,
        checksum=checksum,
        entry_count=entry_count,
    )


def build_access_graph(
    nodes: dict[NodeId, AccessGraphNode],
) -> AccessGraph:
    """Construct a validated AccessGraph. Raises on empty graph or integrity violations."""
    if not nodes:
        raise ValueError("nodes dict must not be empty")

    # Pre-check id/key match before handing to pydantic (clearer error messages)
    for key, node in nodes.items():
        if node.id != key:
            raise ValueError(
                f"Node id '{node.id}' does not match its key '{key}' in the graph"
            )

    return AccessGraph(nodes=nodes, version="1")


def parse_ledger_line(line: str) -> LedgerLine:
    """Parse a single JSONL line into a TrustLedgerEntry or LedgerCheckpoint."""
    if not line or not line.strip():
        raise ValueError("Ledger line must not be empty")

    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    # Discriminate by presence of 'event' field (entry) vs 'checksum' field (checkpoint)
    if "event" in data:
        return TrustLedgerEntry.model_validate(data)
    elif "checksum" in data:
        return LedgerCheckpoint.model_validate(data)
    else:
        raise ValidationError.from_exception_data(
            title="LedgerLine",
            line_errors=[
                {
                    "type": "value_error",
                    "loc": (),
                    "msg": "JSON does not match TrustLedgerEntry or LedgerCheckpoint schema",
                    "input": data,
                    "ctx": {"error": ValueError("unknown ledger line schema")},
                }
            ],
        )


def serialize_ledger_line(entry: Union[TrustLedgerEntry, LedgerCheckpoint]) -> str:
    """Serialize a LedgerLine to a deterministic single-line JSON string."""
    if not isinstance(entry, (TrustLedgerEntry, LedgerCheckpoint)):
        raise TypeError(
            f"Expected TrustLedgerEntry or LedgerCheckpoint, got {type(entry).__name__}"
        )
    return json.dumps(entry.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def create_error_response(
    error_code: str,
    message: str,
    node: str = "",
    field: str = "",
    domain: str = "",
) -> "ErrorResponse":
    """Create a standardized ErrorResponse with context details."""
    from .api import ErrorResponse

    if not error_code:
        raise ValueError("error_code must not be empty")
    if not message:
        raise ValueError("message must not be empty")

    details: dict[str, str] = {}
    if node:
        details["node"] = node
    if field:
        details["field"] = field
    if domain:
        details["domain"] = domain

    return ErrorResponse(error_code=error_code, message=message, details=details)


def validate_canary_fingerprint(fingerprint: str) -> bool:
    """Validate that a canary fingerprint contains an embedded UUID v4.

    Returns True if valid, raises ValueError otherwise.
    """
    if not fingerprint:
        raise ValueError("fingerprint must not be empty")
    if not _UUID_V4_RE.search(fingerprint):
        raise ValueError("fingerprint does not contain a valid UUID v4 segment")
    return True


def score_to_tier(score: TrustScore) -> TrustTier:
    """Convert a raw TrustScore to its display TrustTier.

    Tier boundaries (0.2 increments):
      [0.0, 0.2) -> PROBATIONARY
      [0.2, 0.4) -> LOW
      [0.4, 0.6) -> ESTABLISHED
      [0.6, 0.8) -> HIGH
      [0.8, 1.0] -> TRUSTED
    """
    if score < 0.0 or score > 1.0:
        raise ValueError(f"score must be in [0.0, 1.0], got {score}")

    if score < 0.2:
        return TrustTier.PROBATIONARY
    elif score < 0.4:
        return TrustTier.LOW
    elif score < 0.6:
        return TrustTier.ESTABLISHED
    elif score < 0.8:
        return TrustTier.HIGH
    else:
        return TrustTier.TRUSTED


def classify_field(
    field_name: str,
    rules: list[ClassificationRule],
) -> DataTier:
    """Classify a field name against classification rules. First match wins.

    Returns DataTier.PUBLIC if no rule matches.
    """
    if not field_name:
        raise ValueError("field_name must not be empty")

    for rule in rules:
        if rule.is_regex:
            try:
                if re.search(rule.field_pattern, field_name):
                    return rule.data_tier
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern '{rule.field_pattern}': {exc}") from exc
        else:
            if fnmatch.fnmatch(field_name, rule.field_pattern):
                return rule.data_tier

    return DataTier.PUBLIC
