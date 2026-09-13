"""Access auditor: structural profile computation, slot auditing, observed output auditing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from .classifier import classify_fields
from .errors import (
    AuditInputError,
    ProfileComputationError,
    SchemaWalkError,
    RefResolutionError,
    ClassificationRegistryError,
)
from .models import (
    AccessFinding,
    AccessFindingEvidence,
    ClassificationRegistryEntry,
    ClassifiedField,
    DataTier,
    DeclaredAccess,
    FindingSeverity,
    GateConfig,
    ObservedOutput,
    SlotDecision,
    StructuralProfile,
)
from .walker import walk_response_schema

__all__ = [
    "compute_structural_profile",
    "audit_slot",
    "audit_observed_output",
]


def compute_structural_profile(
    node_id: str,
    adapter_slot_id: str,
    endpoint: str,
    schema: dict[str, Any] | None,
    ref_resolver: Callable[[str], dict[str, Any]],
    registry_entries: list[ClassificationRegistryEntry],
    gate_config: GateConfig,
) -> StructuralProfile:
    """Compose walk + classify into a StructuralProfile. Injects RESTRICTED on incomplete schemas."""
    if not node_id:
        raise ProfileComputationError(node_id="", detail="node_id must not be empty")
    if not adapter_slot_id:
        raise ProfileComputationError(
            node_id=node_id, detail="adapter_slot_id must not be empty"
        )
    if not endpoint:
        raise ProfileComputationError(
            node_id=node_id, detail="endpoint must not be empty"
        )

    try:
        walk_result = walk_response_schema(schema, ref_resolver)
    except (SchemaWalkError, RefResolutionError) as exc:
        raise ProfileComputationError(
            node_id=node_id,
            detail=str(exc),
            endpoint=endpoint,
        ) from exc

    try:
        classification = classify_fields(walk_result.fields, registry_entries)
    except ClassificationRegistryError as exc:
        raise ProfileComputationError(
            node_id=node_id,
            detail=str(exc),
        ) from exc

    tiers = list(classification.tier_set)
    warnings = list(walk_result.warnings)
    schema_complete = len(warnings) == 0

    # Assume-worst: inject RESTRICTED if schema is incomplete
    if not schema_complete and gate_config.assume_worst_on_incomplete:
        if DataTier.RESTRICTED not in tiers:
            tiers.append(DataTier.RESTRICTED)
            tiers.sort()

    now = datetime.now(timezone.utc).isoformat()

    return StructuralProfile(
        node_id=node_id,
        adapter_slot_id=adapter_slot_id,
        endpoint=endpoint,
        tiers=tiers,
        classified_fields=classification.classified_fields,
        warnings=warnings,
        schema_complete=schema_complete,
        computed_at=now,
    )


def audit_slot(
    profile: StructuralProfile,
    declared_access: DeclaredAccess,
    gate_config: GateConfig,
) -> SlotDecision:
    """Compare structural profile vs declared access. Produces C005 findings and BLOCK/ALLOW verdict."""
    if profile.node_id != declared_access.node_id:
        raise AuditInputError(
            f"node_id mismatch: profile has '{profile.node_id}', "
            f"declared has '{declared_access.node_id}'",
            profile_node_id=profile.node_id,
            declared_node_id=declared_access.node_id,
        )

    now = datetime.now(timezone.utc).isoformat()
    findings: list[AccessFinding] = []
    blocking_codes: list[str] = []

    structural_tiers = set(profile.tiers)
    declared_tiers = set(declared_access.declared_read_tiers)
    undeclared_tiers = structural_tiers - declared_tiers

    for tier in sorted(undeclared_tiers):
        tier_fields = sorted(cf.path for cf in profile.classified_fields if cf.tier == tier)
        findings.append(AccessFinding(
            code="C005",
            severity=FindingSeverity.WARNING,
            node_id=profile.node_id,
            adapter_slot_id=profile.adapter_slot_id,
            message=(
                f"DECLARATION_GAP: node '{profile.node_id}' has structural access to "
                f"tier {tier.name} not declared in access graph at endpoint '{profile.endpoint}'"
            ),
            evidence=AccessFindingEvidence(
                structural_tiers=sorted(structural_tiers),
                declared_tiers=sorted(declared_tiers),
                undeclared_tiers=[tier],
                fields_by_undeclared_tier={tier.name: tier_fields},
                endpoint=profile.endpoint,
            ),
            timestamp=now,
        ))
    if undeclared_tiers and "C005" in gate_config.block_on_codes:
        blocking_codes.append("C005")

    # Add INCOMPLETE_SCHEMA findings from profile warnings
    for warning in profile.warnings:
        finding = AccessFinding(
            code="INCOMPLETE_SCHEMA",
            severity=FindingSeverity.INFO,
            node_id=profile.node_id,
            adapter_slot_id=profile.adapter_slot_id,
            message=f"INCOMPLETE_SCHEMA at '{warning.path}': {warning.message}",
            evidence=AccessFindingEvidence(
                structural_tiers=sorted(structural_tiers),
                declared_tiers=sorted(declared_tiers),
                undeclared_tiers=[],
                fields_by_undeclared_tier={},
                endpoint=profile.endpoint,
            ),
            timestamp=now,
        )
        findings.append(finding)

        if "INCOMPLETE_SCHEMA" in gate_config.block_on_codes:
            if "INCOMPLETE_SCHEMA" not in blocking_codes:
                blocking_codes.append("INCOMPLETE_SCHEMA")

    decision = "BLOCK" if blocking_codes else "ALLOW"

    return SlotDecision(
        decision=decision,
        adapter_slot_id=profile.adapter_slot_id,
        node_id=profile.node_id,
        findings=findings,
        blocking_codes=blocking_codes,
        profile=profile,
        decided_at=now,
    )


def audit_observed_output(
    observed: ObservedOutput,
    declared_access: DeclaredAccess,
) -> list[AccessFinding]:
    """Produce FA_A_015 findings when observed output tiers exceed declared reads."""
    if observed.node_id != declared_access.node_id:
        raise AuditInputError(
            f"node_id mismatch: observed has '{observed.node_id}', "
            f"declared has '{declared_access.node_id}'",
            observed_node_id=observed.node_id,
            declared_node_id=declared_access.node_id,
        )

    now = datetime.now(timezone.utc).isoformat()
    findings: list[AccessFinding] = []

    observed_tiers = set(observed.observed_tiers)
    declared_tiers = set(declared_access.declared_read_tiers)
    undeclared_tiers = observed_tiers - declared_tiers

    for tier in sorted(undeclared_tiers):
        # Find fields at this tier
        tier_fields = [
            cf.path for cf in observed.observed_fields if cf.tier == tier
        ]

        fields_by_tier: dict[str, list[str]] = {
            tier.name: sorted(tier_fields),
        }

        evidence = AccessFindingEvidence(
            structural_tiers=sorted(observed_tiers),
            declared_tiers=sorted(declared_tiers),
            undeclared_tiers=[tier],
            fields_by_undeclared_tier=fields_by_tier,
            endpoint="",
        )

        field_list = ", ".join(sorted(tier_fields)) if tier_fields else "unknown"
        finding = AccessFinding(
            code="FA_A_015",
            severity=FindingSeverity.ERROR,
            node_id=observed.node_id,
            adapter_slot_id=observed.adapter_slot_id,
            message=(
                f"ACCESS_VIOLATION: node '{observed.node_id}' observed output "
                f"includes tier {tier.name} (fields: {field_list}) which is not "
                f"in declared reads"
            ),
            evidence=evidence,
            timestamp=now,
        )
        findings.append(finding)

    return findings
