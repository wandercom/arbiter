"""Consistency analysis: compare adapter observations against node claims."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .models import (
    AdapterObservation,
    AnalysisPair,
    AnalysisPairList,
    ConsistencyAnalysisError,
    ConsistencyFinding,
    ConsistencyFindingList,
    ConsistencyOutcome,
    FindingSeverity,
    NodeAuditClaim,
)

__all__ = ["analyze_span", "analyze_batch"]

# Dot-notation field name pattern: segments of word chars separated by dots.
# Each segment must be non-empty and consist of alphanumeric + underscore chars.
_FIELD_RE = re.compile(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$")

# Threshold: unexplained field count at or above this is HIGH severity
_HIGH_THRESHOLD = 3


def _now_utc_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _validate_fields(
    fields: frozenset[str],
    source_label: str,
    node_id: str,
    span_id: str,
) -> None:
    """Validate that all field names are valid dot-notation identifiers."""
    for field in fields:
        if not _FIELD_RE.match(field):
            raise ConsistencyAnalysisError(
                node_id=node_id,
                span_id=span_id,
                detail=(
                    f"Malformed field name in {source_label} for node "
                    f"{node_id}, span {span_id}: {field}"
                ),
            )


def _compute_severity(
    outcome: ConsistencyOutcome,
    unexplained_count: int,
    overclaimed_count: int,
) -> FindingSeverity:
    """Deterministic severity from outcome and field counts."""
    if outcome == ConsistencyOutcome.CONSISTENT:
        return FindingSeverity.NONE
    if outcome == ConsistencyOutcome.MISSING_CLAIM:
        return FindingSeverity.HIGH
    if outcome == ConsistencyOutcome.MISSING_OBSERVATION:
        return FindingSeverity.LOW
    # INCONSISTENT
    if unexplained_count == 0:
        # overclaim only
        return FindingSeverity.LOW
    if unexplained_count >= _HIGH_THRESHOLD:
        return FindingSeverity.HIGH
    return FindingSeverity.MEDIUM


def analyze_span(
    observation: AdapterObservation | None,
    claim: NodeAuditClaim | None,
) -> ConsistencyFinding:
    """Compare a single adapter observation against a single node audit claim.

    At least one of *observation* or *claim* must be non-None.
    """
    if observation is None and claim is None:
        raise ConsistencyAnalysisError(
            node_id="",
            span_id="",
            detail=(
                "analyze_span requires at least one of observation or "
                "claim; both were None"
            ),
        )

    now = _now_utc_iso()

    # --- MISSING_OBSERVATION ---
    if observation is None and claim is not None:
        _validate_fields(claim.claimed_fields, "claim", claim.node_id, claim.span_id)
        return ConsistencyFinding(
            schema_version=1,
            node_id=claim.node_id,
            span_id=claim.span_id,
            trace_id=claim.trace_id,
            outcome=ConsistencyOutcome.MISSING_OBSERVATION,
            severity=_compute_severity(
                ConsistencyOutcome.MISSING_OBSERVATION, 0, len(claim.claimed_fields)
            ),
            observed_fields=frozenset(),
            claimed_fields=claim.claimed_fields,
            unexplained_fields=frozenset(),
            overclaimed_fields=claim.claimed_fields,
            analyzed_at=now,
            details="Observation missing; claim present",
        )

    # --- MISSING_CLAIM ---
    if observation is not None and claim is None:
        _validate_fields(
            observation.observed_fields, "observation",
            observation.node_id, observation.span_id,
        )
        return ConsistencyFinding(
            schema_version=1,
            node_id=observation.node_id,
            span_id=observation.span_id,
            trace_id=observation.trace_id,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            severity=FindingSeverity.HIGH,
            observed_fields=observation.observed_fields,
            claimed_fields=frozenset(),
            unexplained_fields=observation.observed_fields,
            overclaimed_fields=frozenset(),
            analyzed_at=now,
            details="Claim missing; observation present",
        )

    # --- Both present ---
    assert observation is not None and claim is not None

    # Validate IDs match
    if observation.span_id != claim.span_id:
        raise ConsistencyAnalysisError(
            node_id=observation.node_id,
            span_id=observation.span_id,
            detail=(
                f"span_id mismatch between observation "
                f"({observation.span_id}) and claim ({claim.span_id})"
            ),
        )
    if observation.node_id != claim.node_id:
        raise ConsistencyAnalysisError(
            node_id=observation.node_id,
            span_id=observation.span_id,
            detail=(
                f"node_id mismatch between observation "
                f"({observation.node_id}) and claim ({claim.node_id})"
            ),
        )
    if observation.trace_id != claim.trace_id:
        raise ConsistencyAnalysisError(
            node_id=observation.node_id,
            span_id=observation.span_id,
            detail=(
                f"trace_id mismatch between observation "
                f"({observation.trace_id}) and claim ({claim.trace_id})"
            ),
        )

    _validate_fields(
        observation.observed_fields, "observation",
        observation.node_id, observation.span_id,
    )
    _validate_fields(
        claim.claimed_fields, "claim",
        claim.node_id, claim.span_id,
    )

    observed_set = observation.observed_fields
    claimed_set = claim.claimed_fields

    unexplained = observed_set - claimed_set
    overclaimed = claimed_set - observed_set

    if not unexplained and not overclaimed:
        outcome = ConsistencyOutcome.CONSISTENT
    else:
        outcome = ConsistencyOutcome.INCONSISTENT

    severity = _compute_severity(outcome, len(unexplained), len(overclaimed))

    return ConsistencyFinding(
        schema_version=1,
        node_id=observation.node_id,
        span_id=observation.span_id,
        trace_id=observation.trace_id,
        outcome=outcome,
        severity=severity,
        observed_fields=observed_set,
        claimed_fields=claimed_set,
        unexplained_fields=unexplained,
        overclaimed_fields=overclaimed,
        analyzed_at=now,
        details=(
            f"Unexplained fields: {', '.join(sorted(unexplained)) or 'none'}; "
            f"overclaimed fields: {', '.join(sorted(overclaimed)) or 'none'}"
            if outcome == ConsistencyOutcome.INCONSISTENT else None
        ),
    )


def analyze_batch(
    pairs: AnalysisPairList,
) -> ConsistencyFindingList:
    """Analyze a sequence of observation/claim pairs.

    Returns one ConsistencyFinding per pair. Not atomic: error on any
    pair propagates immediately.
    """
    if not pairs:
        raise ConsistencyAnalysisError(
            node_id="",
            span_id="",
            detail="analyze_batch called with empty pairs sequence",
        )

    results: ConsistencyFindingList = []
    for i, pair in enumerate(pairs):
        if pair.observation is None and pair.claim is None:
            raise ConsistencyAnalysisError(
                node_id="",
                span_id="",
                detail=(
                    f"Pair at index {i} has both observation and "
                    f"claim as None"
                ),
            )
        try:
            results.append(analyze_span(pair.observation, pair.claim))
        except ConsistencyAnalysisError as exc:
            raise ConsistencyAnalysisError(
                node_id=exc.node_id,
                span_id=exc.span_id,
                detail=f"ID mismatch in pair at index {i}: {exc.detail}",
            ) from exc

    return results
