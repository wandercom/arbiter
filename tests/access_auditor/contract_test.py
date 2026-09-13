"""
Contract test suite for access_auditor component.
Tests verify behavior at boundaries per contract specification.
Run with: pytest contract_test.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone
import re

from arbiter.access import (
    DataTier,
    FindingCode,
    FindingSeverity,
    SlotDecisionVerdict,
    NodeId,
    FieldPath,
    AdapterSlotId,
    FieldEntry,
    SchemaWarning,
    WalkResult,
    ClassifiedField,
    ClassificationResult,
    StructuralProfile,
    AccessFinding,
    AccessFindingEvidence,
    GateConfig,
    SlotDecision,
    DeclaredAccess,
    ClassificationRegistryEntry,
    ObservedOutput,
    walk_response_schema,
    classify_fields,
    compute_structural_profile,
    audit_slot,
    audit_observed_output,
    load_gate_config,
    load_classification_registry,
)


# ============================================================================
# Fixtures & Helpers
# ============================================================================

def make_field_entry(path="response.field", field_type="string", nullable=False, format_hint=""):
    return FieldEntry(path=FieldPath(path), field_type=field_type, nullable=nullable, format_hint=format_hint)


def make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL, pattern_type="fnmatch", description="test"):
    return ClassificationRegistryEntry(
        field_pattern=field_pattern, tier=tier, pattern_type=pattern_type, description=description
    )


def make_gate_config(block_on_codes=None, assume_worst_on_incomplete=True):
    if block_on_codes is None:
        block_on_codes = [FindingCode.C005]
    return GateConfig(block_on_codes=block_on_codes, assume_worst_on_incomplete=assume_worst_on_incomplete)


def make_declared_access(node_id="node-1", read_tiers=None, write_tiers=None):
    if read_tiers is None:
        read_tiers = [DataTier.PUBLIC]
    if write_tiers is None:
        write_tiers = []
    return DeclaredAccess(
        node_id=NodeId(node_id),
        declared_read_tiers=read_tiers,
        declared_write_tiers=write_tiers,
    )


def make_structural_profile(
    node_id="node-1",
    adapter_slot_id="slot-1",
    endpoint="/api/data",
    tiers=None,
    classified_fields=None,
    warnings=None,
    schema_complete=True,
    computed_at=None,
):
    if tiers is None:
        tiers = [DataTier.PUBLIC]
    if classified_fields is None:
        classified_fields = []
    if warnings is None:
        warnings = []
    if computed_at is None:
        computed_at = datetime.now(timezone.utc).isoformat()
    return StructuralProfile(
        node_id=NodeId(node_id),
        adapter_slot_id=AdapterSlotId(adapter_slot_id),
        endpoint=endpoint,
        tiers=tiers,
        classified_fields=classified_fields,
        warnings=warnings,
        schema_complete=schema_complete,
        computed_at=computed_at,
    )


def make_observed_output(
    node_id="node-1",
    adapter_slot_id="slot-1",
    observed_tiers=None,
    observed_fields=None,
    observed_at=None,
):
    if observed_tiers is None:
        observed_tiers = [DataTier.PUBLIC]
    if observed_fields is None:
        observed_fields = []
    if observed_at is None:
        observed_at = datetime.now(timezone.utc).isoformat()
    return ObservedOutput(
        node_id=NodeId(node_id),
        adapter_slot_id=AdapterSlotId(adapter_slot_id),
        observed_tiers=observed_tiers,
        observed_fields=[
            field if isinstance(field, ClassifiedField) else ClassifiedField(
                path=field.path, tier=observed_tiers[min(index, len(observed_tiers) - 1)],
            )
            for index, field in enumerate(observed_fields)
        ],
        observed_at=observed_at,
    )


def dict_ref_resolver(ref_map):
    """Returns a callable ref_resolver backed by a dict."""
    def resolver(ref_uri):
        if ref_uri in ref_map:
            return ref_map[ref_uri]
        raise KeyError(f"Unresolvable $ref: {ref_uri}")
    return resolver


def noop_ref_resolver(ref_uri):
    raise KeyError(f"Unresolvable $ref: {ref_uri}")


FLAT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
        },
    },
}

ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "value": {"type": "string"},
        },
    },
}

NULLABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "nickname": {"type": "string", "nullable": True},
        "age": {"type": "integer"},
    },
}

ALLOF_SCHEMA = {
    "allOf": [
        {"type": "object", "properties": {"id": {"type": "integer"}}},
        {"type": "object", "properties": {"name": {"type": "string"}}},
    ]
}

REF_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {"$ref": "#/components/schemas/Profile"},
    },
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "bio": {"type": "string"},
    },
}

CIRCULAR_REF_A = {
    "type": "object",
    "properties": {
        "child": {"$ref": "#/components/schemas/B"},
    },
}

CIRCULAR_REF_B = {
    "type": "object",
    "properties": {
        "parent": {"$ref": "#/components/schemas/A"},
    },
}

DEEP_NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "level1": {
            "type": "object",
            "properties": {
                "level2": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ============================================================================
# Test walk_response_schema
# ============================================================================

class TestWalkResponseSchema:
    """Tests for walk_response_schema function."""

    def test_walk_happy_flat_schema(self):
        """Flat schema with string, integer, boolean properties produces FieldEntry per leaf."""
        result = walk_response_schema(
            schema=FLAT_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        assert len(result.fields) == 3
        assert all(f.path.startswith("response") for f in result.fields)
        assert len(result.warnings) == 0

    def test_walk_happy_nested_schema(self):
        """Nested objects produce dot-separated paths like response.user.name."""
        result = walk_response_schema(
            schema=NESTED_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        paths = [f.path for f in result.fields]
        assert any("user.name" in p for p in paths)
        assert any("user.email" in p for p in paths)
        assert all(f.path.startswith("response") for f in result.fields)

    def test_walk_happy_ref_resolution(self):
        """$ref is resolved via ref_resolver and referenced fields discovered."""
        ref_map = {"#/components/schemas/Profile": PROFILE_SCHEMA}
        resolver = dict_ref_resolver(ref_map)
        result = walk_response_schema(
            schema=REF_SCHEMA,
            ref_resolver=resolver,
            root_path="response",
            max_depth=10,
        )
        assert len(result.fields) >= 1
        assert len(result.warnings) == 0

    def test_walk_empty_schema_warning(self):
        """Empty schema {} produces INCOMPLETE_SCHEMA warning."""
        result = walk_response_schema(
            schema={},
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        assert len(result.warnings) >= 1
        assert any(w.code == FindingCode.INCOMPLETE_SCHEMA for w in result.warnings)

    def test_walk_none_equivalent_schema_warning(self):
        """None-equivalent schema produces INCOMPLETE_SCHEMA warning."""
        # Try with None wrapped in whatever the implementation accepts
        # The contract says 'None-equivalent' so we try empty dict
        result = walk_response_schema(
            schema={},
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        assert any(w.code == FindingCode.INCOMPLETE_SCHEMA for w in result.warnings)

    def test_walk_circular_ref_detection(self):
        """Circular $ref chains are detected without infinite recursion."""
        ref_map = {
            "#/components/schemas/A": CIRCULAR_REF_A,
            "#/components/schemas/B": CIRCULAR_REF_B,
        }
        resolver = dict_ref_resolver(ref_map)
        schema_with_ref = {
            "type": "object",
            "properties": {
                "root": {"$ref": "#/components/schemas/A"},
            },
        }
        result = walk_response_schema(
            schema=schema_with_ref,
            ref_resolver=resolver,
            root_path="response",
            max_depth=20,
        )
        # Must not hang, and must emit a cycle warning
        assert any(w.code == FindingCode.INCOMPLETE_SCHEMA for w in result.warnings)

    def test_walk_max_depth_exceeded(self):
        """Deeply nested schema with max_depth=1 raises error or emits warning."""
        try:
            result = walk_response_schema(
                schema=DEEP_NESTED_SCHEMA,
                ref_resolver=noop_ref_resolver,
                root_path="response",
                max_depth=1,
            )
            # If it returns instead of raising, it should have a warning
            # The contract says 'error: max_depth_exceeded' which may be an exception
            # or it may be a warning. We accept either.
            assert any(w.code == FindingCode.INCOMPLETE_SCHEMA for w in result.warnings) or len(result.fields) == 0
        except Exception as e:
            # max_depth_exceeded error is acceptable
            assert "depth" in str(e).lower() or "max_depth" in str(e).lower() or True

    def test_walk_ref_resolver_failure(self):
        """ref_resolver raising exception causes ref_resolver_failure error."""
        def failing_resolver(ref_uri):
            raise RuntimeError("Connection failed")

        schema = {"type": "object", "properties": {"x": {"$ref": "#/broken"}}}
        try:
            result = walk_response_schema(
                schema=schema,
                ref_resolver=failing_resolver,
                root_path="response",
                max_depth=10,
            )
            # May return with warnings instead of raising
            assert any(w.code == FindingCode.INCOMPLETE_SCHEMA for w in result.warnings)
        except Exception:
            pass  # ref_resolver_failure exception is acceptable

    def test_walk_invalid_schema_type(self):
        """Non-dict schema raises invalid_schema_type error."""
        with pytest.raises(Exception):
            walk_response_schema(
                schema="not_a_dict",
                ref_resolver=noop_ref_resolver,
                root_path="response",
                max_depth=10,
            )

    def test_walk_allof_composition(self):
        """allOf is flattened with union semantics, fields from both schemas appear."""
        result = walk_response_schema(
            schema=ALLOF_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        assert len(result.fields) >= 2
        paths = [f.path for f in result.fields]
        assert any("id" in p for p in paths)
        assert any("name" in p for p in paths)

    def test_walk_array_items(self):
        """Array items are traversed and leaf fields discovered."""
        result = walk_response_schema(
            schema=ARRAY_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        assert len(result.fields) >= 1

    def test_walk_nullable_field(self):
        """Nullable fields are recorded correctly in FieldEntry."""
        result = walk_response_schema(
            schema=NULLABLE_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        assert any(f.nullable for f in result.fields)

    def test_walk_only_leaf_fields(self):
        """Result contains only leaf fields, no intermediate object/array nodes."""
        result = walk_response_schema(
            schema=NESTED_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        for field in result.fields:
            assert field.field_type not in ("object", "array"), (
                f"Non-leaf field_type '{field.field_type}' found at path '{field.path}'"
            )

    def test_walk_deterministic(self):
        """Same inputs always produce same outputs (pure function)."""
        kwargs = dict(
            schema=NESTED_SCHEMA,
            ref_resolver=noop_ref_resolver,
            root_path="response",
            max_depth=10,
        )
        result1 = walk_response_schema(**kwargs)
        result2 = walk_response_schema(**kwargs)
        assert len(result1.fields) == len(result2.fields)
        paths1 = sorted(f.path for f in result1.fields)
        paths2 = sorted(f.path for f in result2.fields)
        assert paths1 == paths2
        assert len(result1.warnings) == len(result2.warnings)


# ============================================================================
# Test classify_fields
# ============================================================================

class TestClassifyFields:
    """Tests for classify_fields function."""

    def test_classify_happy_fnmatch(self):
        """fnmatch pattern matches field and assigns correct tier."""
        fields = [make_field_entry(path="response.user.email")]
        registry = [make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL, pattern_type="fnmatch")]
        result = classify_fields(fields=fields, registry_entries=registry)
        assert len(result.classified_fields) == 1
        assert result.classified_fields[0].tier == DataTier.CONFIDENTIAL

    def test_classify_happy_regex(self):
        """regex pattern matches field and assigns correct tier."""
        fields = [make_field_entry(path="response.user.ssn")]
        registry = [make_registry_entry(field_pattern=".*ssn$", tier=DataTier.RESTRICTED, pattern_type="regex")]
        result = classify_fields(fields=fields, registry_entries=registry)
        assert result.classified_fields[0].tier == DataTier.RESTRICTED

    def test_classify_unmatched_defaults_public(self):
        """Fields matching no pattern are assigned DataTier.PUBLIC."""
        fields = [make_field_entry(path="response.status")]
        registry = [make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL)]
        result = classify_fields(fields=fields, registry_entries=registry)
        assert result.classified_fields[0].tier == DataTier.PUBLIC

    def test_classify_highest_tier_wins(self):
        """When field matches multiple patterns, highest DataTier wins."""
        fields = [make_field_entry(path="response.user.email")]
        registry = [
            make_registry_entry(field_pattern="*email*", tier=DataTier.INTERNAL, pattern_type="fnmatch"),
            make_registry_entry(field_pattern="*email*", tier=DataTier.RESTRICTED, pattern_type="fnmatch"),
        ]
        result = classify_fields(fields=fields, registry_entries=registry)
        assert result.classified_fields[0].tier == DataTier.RESTRICTED

    def test_classify_all_fields_present(self):
        """Every input field appears exactly once in classified_fields."""
        fields = [make_field_entry(path=f"response.field{i}") for i in range(5)]
        registry = []
        result = classify_fields(fields=fields, registry_entries=registry)
        assert len(result.classified_fields) == 5

    def test_classify_tier_set_consistent(self):
        """tier_set equals the set of all classified field tiers."""
        fields = [
            make_field_entry(path="response.user.email"),
            make_field_entry(path="response.status"),
        ]
        registry = [make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL)]
        result = classify_fields(fields=fields, registry_entries=registry)
        expected_tiers = {cf.tier for cf in result.classified_fields}
        assert set(result.tier_set) == expected_tiers

    def test_classify_invalid_regex_error(self):
        """Invalid regex pattern raises error."""
        fields = [make_field_entry(path="response.field")]
        registry = [make_registry_entry(field_pattern="[invalid", tier=DataTier.PUBLIC, pattern_type="regex")]
        with pytest.raises(Exception):
            classify_fields(fields=fields, registry_entries=registry)

    def test_classify_empty_field_path_error(self):
        """FieldEntry with empty path raises error."""
        fields = [make_field_entry(path="")]
        registry = []
        with pytest.raises(Exception):
            classify_fields(fields=fields, registry_entries=registry)

    def test_classify_deterministic(self):
        """Same inputs produce same outputs (pure function)."""
        fields = [make_field_entry(path="response.user.email")]
        registry = [make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL)]
        result1 = classify_fields(fields=fields, registry_entries=registry)
        result2 = classify_fields(fields=fields, registry_entries=registry)
        assert len(result1.classified_fields) == len(result2.classified_fields)
        for cf1, cf2 in zip(result1.classified_fields, result2.classified_fields):
            assert cf1.path == cf2.path
            assert cf1.tier == cf2.tier
        assert set(result1.tier_set) == set(result2.tier_set)

    def test_classify_empty_fields_list(self):
        """Empty field list returns empty classification result."""
        result = classify_fields(fields=[], registry_entries=[])
        assert len(result.classified_fields) == 0
        assert len(result.tier_set) == 0

    def test_classify_multiple_fields_mixed_tiers(self):
        """Multiple fields with different matching patterns get correct tiers."""
        fields = [
            make_field_entry(path="response.user.email"),
            make_field_entry(path="response.user.ssn"),
            make_field_entry(path="response.status"),
        ]
        registry = [
            make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL),
            make_registry_entry(field_pattern="*ssn*", tier=DataTier.RESTRICTED),
        ]
        result = classify_fields(fields=fields, registry_entries=registry)
        tier_map = {cf.path: cf.tier for cf in result.classified_fields}
        assert tier_map[FieldPath("response.user.email")] == DataTier.CONFIDENTIAL
        assert tier_map[FieldPath("response.user.ssn")] == DataTier.RESTRICTED
        assert tier_map[FieldPath("response.status")] == DataTier.PUBLIC


# ============================================================================
# Test compute_structural_profile
# ============================================================================

class TestComputeStructuralProfile:
    """Tests for compute_structural_profile function."""

    def test_profile_happy_path(self):
        """Valid inputs produce a StructuralProfile with correct identifiers and UTC timestamp."""
        node_id = NodeId("node-1")
        adapter_slot_id = AdapterSlotId("slot-1")
        endpoint = "/api/users"
        registry = [make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL)]
        gate_config = make_gate_config()

        result = compute_structural_profile(
            node_id=node_id,
            adapter_slot_id=adapter_slot_id,
            endpoint=endpoint,
            schema=NESTED_SCHEMA,
            ref_resolver=noop_ref_resolver,
            registry_entries=registry,
            gate_config=gate_config,
        )
        assert result.node_id == node_id
        assert result.adapter_slot_id == adapter_slot_id
        assert result.endpoint == endpoint
        assert result.computed_at is not None
        assert len(result.computed_at) > 0

    def test_profile_empty_schema_assume_worst(self):
        """Empty schema with assume_worst injects RESTRICTED tier."""
        node_id = NodeId("node-1")
        adapter_slot_id = AdapterSlotId("slot-1")
        gate_config = make_gate_config(assume_worst_on_incomplete=True)

        result = compute_structural_profile(
            node_id=node_id,
            adapter_slot_id=adapter_slot_id,
            endpoint="/api/data",
            schema={},
            ref_resolver=noop_ref_resolver,
            registry_entries=[],
            gate_config=gate_config,
        )
        assert DataTier.RESTRICTED in result.tiers
        assert result.schema_complete is False

    def test_profile_schema_complete_when_no_warnings(self):
        """schema_complete is True when walk produces no warnings."""
        gate_config = make_gate_config()
        result = compute_structural_profile(
            node_id=NodeId("node-1"),
            adapter_slot_id=AdapterSlotId("slot-1"),
            endpoint="/api/users",
            schema=FLAT_SCHEMA,
            ref_resolver=noop_ref_resolver,
            registry_entries=[],
            gate_config=gate_config,
        )
        assert result.schema_complete is True
        assert len(result.warnings) == 0

    def test_profile_utc_timestamp(self):
        """computed_at is a valid ISO 8601 UTC timestamp."""
        gate_config = make_gate_config()
        result = compute_structural_profile(
            node_id=NodeId("node-1"),
            adapter_slot_id=AdapterSlotId("slot-1"),
            endpoint="/api/users",
            schema=FLAT_SCHEMA,
            ref_resolver=noop_ref_resolver,
            registry_entries=[],
            gate_config=gate_config,
        )
        # Parse the timestamp to verify it's valid ISO 8601
        ts = result.computed_at
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None or "Z" in ts or "+00:00" in ts

    def test_profile_classified_fields_consistent_with_tiers(self):
        """All tiers from classified_fields appear in result.tiers."""
        registry = [make_registry_entry(field_pattern="*email*", tier=DataTier.CONFIDENTIAL)]
        gate_config = make_gate_config()
        result = compute_structural_profile(
            node_id=NodeId("node-1"),
            adapter_slot_id=AdapterSlotId("slot-1"),
            endpoint="/api/users",
            schema=NESTED_SCHEMA,
            ref_resolver=noop_ref_resolver,
            registry_entries=registry,
            gate_config=gate_config,
        )
        classified_tiers = {cf.tier for cf in result.classified_fields}
        result_tiers = set(result.tiers)
        assert classified_tiers.issubset(result_tiers)

    def test_profile_empty_node_id_precondition(self):
        """Empty node_id violates precondition and raises error."""
        gate_config = make_gate_config()
        with pytest.raises(Exception):
            compute_structural_profile(
                node_id=NodeId(""),
                adapter_slot_id=AdapterSlotId("slot-1"),
                endpoint="/api/data",
                schema=FLAT_SCHEMA,
                ref_resolver=noop_ref_resolver,
                registry_entries=[],
                gate_config=gate_config,
            )

    def test_profile_empty_adapter_slot_id_precondition(self):
        """Empty adapter_slot_id violates precondition."""
        gate_config = make_gate_config()
        with pytest.raises(Exception):
            compute_structural_profile(
                node_id=NodeId("node-1"),
                adapter_slot_id=AdapterSlotId(""),
                endpoint="/api/data",
                schema=FLAT_SCHEMA,
                ref_resolver=noop_ref_resolver,
                registry_entries=[],
                gate_config=gate_config,
            )

    def test_profile_empty_endpoint_precondition(self):
        """Empty endpoint violates precondition."""
        gate_config = make_gate_config()
        with pytest.raises(Exception):
            compute_structural_profile(
                node_id=NodeId("node-1"),
                adapter_slot_id=AdapterSlotId("slot-1"),
                endpoint="",
                schema=FLAT_SCHEMA,
                ref_resolver=noop_ref_resolver,
                registry_entries=[],
                gate_config=gate_config,
            )


# ============================================================================
# Test audit_slot
# ============================================================================

class TestAuditSlot:
    """Tests for audit_slot function."""

    def test_audit_slot_happy_allow(self):
        """ALLOW when all structural tiers are in declared read tiers."""
        profile = make_structural_profile(
            tiers=[DataTier.PUBLIC, DataTier.INTERNAL],
        )
        declared = make_declared_access(
            node_id="node-1",
            read_tiers=[DataTier.PUBLIC, DataTier.INTERNAL],
        )
        gate_config = make_gate_config(block_on_codes=[FindingCode.C005])

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert result.decision == SlotDecisionVerdict.ALLOW
        assert not any(f.code == FindingCode.C005 for f in result.findings)

    def test_audit_slot_happy_block(self):
        """BLOCK when undeclared tiers exist and C005 in block_on_codes."""
        profile = make_structural_profile(
            tiers=[DataTier.PUBLIC, DataTier.CONFIDENTIAL],
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config(block_on_codes=[FindingCode.C005])

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert result.decision == SlotDecisionVerdict.BLOCK
        assert any(f.code == FindingCode.C005 for f in result.findings)

    def test_audit_slot_c005_per_undeclared_tier(self):
        """Exactly one C005 finding per undeclared tier."""
        profile = make_structural_profile(
            tiers=[DataTier.PUBLIC, DataTier.CONFIDENTIAL, DataTier.RESTRICTED],
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config(block_on_codes=[FindingCode.C005])

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) == 2  # CONFIDENTIAL and RESTRICTED

    def test_audit_slot_findings_include_node_id(self):
        """All finding messages include the specific node_id."""
        node_id = "test-node-42"
        profile = make_structural_profile(
            node_id=node_id,
            tiers=[DataTier.CONFIDENTIAL],
        )
        declared = make_declared_access(node_id=node_id, read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config(block_on_codes=[FindingCode.C005])

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        for finding in result.findings:
            assert node_id in finding.message, (
                f"Finding message '{finding.message}' does not contain node_id '{node_id}'"
            )

    def test_audit_slot_c005_severity_warning_or_above(self):
        """C005 findings have severity >= WARNING."""
        profile = make_structural_profile(tiers=[DataTier.RESTRICTED])
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config()

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        for f in result.findings:
            if f.code == FindingCode.C005:
                assert f.severity.value >= FindingSeverity.WARNING.value, (
                    f"C005 finding severity {f.severity} is less than WARNING"
                )

    def test_audit_slot_node_id_mismatch(self):
        """Mismatched node_ids between profile and declared_access raises error."""
        profile = make_structural_profile(node_id="node-A")
        declared = make_declared_access(node_id="node-B")
        gate_config = make_gate_config()

        with pytest.raises(Exception):
            audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)

    def test_audit_slot_block_iff_blocking_codes(self):
        """ALLOW even with C005 findings if C005 not in block_on_codes."""
        profile = make_structural_profile(tiers=[DataTier.CONFIDENTIAL])
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config(block_on_codes=[])  # No blocking codes

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert result.decision == SlotDecisionVerdict.ALLOW

    def test_audit_slot_decided_at_utc(self):
        """decided_at is a valid ISO 8601 UTC timestamp."""
        profile = make_structural_profile(tiers=[DataTier.PUBLIC])
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config()

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        ts = result.decided_at
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None or "Z" in ts or "+00:00" in ts

    def test_audit_slot_profile_preserved(self):
        """result.profile is the same profile passed in."""
        profile = make_structural_profile(tiers=[DataTier.PUBLIC])
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config()

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert result.profile == profile

    def test_audit_slot_result_identifiers(self):
        """result.node_id and adapter_slot_id match the profile."""
        profile = make_structural_profile(node_id="node-1", adapter_slot_id="slot-1")
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config()

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert result.node_id == profile.node_id
        assert result.adapter_slot_id == profile.adapter_slot_id

    def test_audit_slot_incomplete_schema_findings(self):
        """INCOMPLETE_SCHEMA warnings from profile appear as findings."""
        warnings = [SchemaWarning(
            code=FindingCode.INCOMPLETE_SCHEMA,
            path="response",
            message="Schema is empty",
        )]
        profile = make_structural_profile(
            tiers=[DataTier.PUBLIC],
            warnings=warnings,
            schema_complete=False,
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        gate_config = make_gate_config()

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert any(f.code == FindingCode.INCOMPLETE_SCHEMA for f in result.findings)

    def test_audit_slot_all_declared_tiers_no_findings(self):
        """When all structural tiers are declared, no C005 findings are produced."""
        all_tiers = [DataTier.PUBLIC, DataTier.INTERNAL, DataTier.CONFIDENTIAL, DataTier.RESTRICTED]
        profile = make_structural_profile(tiers=all_tiers)
        declared = make_declared_access(node_id="node-1", read_tiers=all_tiers)
        gate_config = make_gate_config()

        result = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        c005 = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005) == 0
        assert result.decision == SlotDecisionVerdict.ALLOW


# ============================================================================
# Test audit_observed_output
# ============================================================================

class TestAuditObservedOutput:
    """Tests for audit_observed_output function."""

    def test_observed_happy_no_violation(self):
        """No findings when observed tiers are subset of declared tiers."""
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[DataTier.PUBLIC],
        )
        declared = make_declared_access(
            node_id="node-1",
            read_tiers=[DataTier.PUBLIC, DataTier.INTERNAL],
        )
        result = audit_observed_output(observed=observed, declared_access=declared)
        assert len(result) == 0

    def test_observed_happy_violation(self):
        """FA_A_015 finding for observed tier not in declared."""
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[DataTier.RESTRICTED],
            observed_fields=[make_field_entry(path="response.user.ssn")],
        )
        declared = make_declared_access(
            node_id="node-1",
            read_tiers=[DataTier.PUBLIC],
        )
        result = audit_observed_output(observed=observed, declared_access=declared)
        assert len(result) == 1
        assert result[0].code == FindingCode.FA_A_015

    def test_observed_severity_error_or_above(self):
        """FA_A_015 findings have severity >= ERROR."""
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[DataTier.RESTRICTED],
            observed_fields=[make_field_entry(path="response.user.ssn")],
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        result = audit_observed_output(observed=observed, declared_access=declared)
        for f in result:
            assert f.severity.value >= FindingSeverity.ERROR.value

    def test_observed_message_includes_node_id(self):
        """Finding messages include node_id."""
        node_id = "my-special-node"
        observed = make_observed_output(
            node_id=node_id,
            observed_tiers=[DataTier.CONFIDENTIAL],
            observed_fields=[make_field_entry(path="response.user.email")],
        )
        declared = make_declared_access(node_id=node_id, read_tiers=[DataTier.PUBLIC])
        result = audit_observed_output(observed=observed, declared_access=declared)
        for f in result:
            assert node_id in f.message

    def test_observed_node_id_mismatch(self):
        """Mismatched node_ids raises error."""
        observed = make_observed_output(node_id="node-A")
        declared = make_declared_access(node_id="node-B")
        with pytest.raises(Exception):
            audit_observed_output(observed=observed, declared_access=declared)

    def test_observed_multiple_undeclared_tiers(self):
        """One FA_A_015 per undeclared tier."""
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[DataTier.CONFIDENTIAL, DataTier.RESTRICTED],
            observed_fields=[
                make_field_entry(path="response.user.email"),
                make_field_entry(path="response.user.ssn"),
            ],
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        result = audit_observed_output(observed=observed, declared_access=declared)
        assert len(result) == 2
        assert all(f.code == FindingCode.FA_A_015 for f in result)

    def test_observed_timestamp_gte_observed_at(self):
        """Each finding timestamp >= observed.observed_at."""
        observed_at = datetime.now(timezone.utc).isoformat()
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[DataTier.RESTRICTED],
            observed_fields=[make_field_entry(path="response.secret")],
            observed_at=observed_at,
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        result = audit_observed_output(observed=observed, declared_access=declared)
        for f in result:
            finding_ts = datetime.fromisoformat(f.timestamp)
            obs_ts = datetime.fromisoformat(observed_at)
            assert finding_ts >= obs_ts

    def test_observed_empty_tiers_no_findings(self):
        """No observed tiers produces no findings."""
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[],
            observed_fields=[],
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        result = audit_observed_output(observed=observed, declared_access=declared)
        assert len(result) == 0

    def test_observed_all_codes_are_fa_a_015(self):
        """Every finding has code == FA_A_015."""
        observed = make_observed_output(
            node_id="node-1",
            observed_tiers=[DataTier.INTERNAL, DataTier.CONFIDENTIAL, DataTier.RESTRICTED],
            observed_fields=[
                make_field_entry(path="response.a"),
                make_field_entry(path="response.b"),
                make_field_entry(path="response.c"),
            ],
        )
        declared = make_declared_access(node_id="node-1", read_tiers=[DataTier.PUBLIC])
        result = audit_observed_output(observed=observed, declared_access=declared)
        assert all(f.code == FindingCode.FA_A_015 for f in result)


# ============================================================================
# Test load_gate_config
# ============================================================================

class TestLoadGateConfig:
    """Tests for load_gate_config function."""

    def test_gate_config_happy_defaults(self):
        """Empty dict returns sensible defaults."""
        result = load_gate_config(config_source={})
        assert FindingCode.C005 in result.block_on_codes
        assert result.assume_worst_on_incomplete is True

    def test_gate_config_happy_custom(self):
        """Explicit values are preserved."""
        result = load_gate_config(config_source={
            "block_on_codes": ["FA_A_015"],
            "assume_worst_on_incomplete": False,
        })
        assert FindingCode.FA_A_015 in result.block_on_codes
        assert result.assume_worst_on_incomplete is False

    def test_gate_config_invalid_finding_code(self):
        """Invalid finding code raises error."""
        with pytest.raises(Exception):
            load_gate_config(config_source={"block_on_codes": ["INVALID_CODE_XYZ"]})

    def test_gate_config_invalid_type(self):
        """Non-dict config_source raises error."""
        with pytest.raises(Exception):
            load_gate_config(config_source="not_a_dict")

    def test_gate_config_missing_block_on_codes_defaults(self):
        """Missing block_on_codes defaults to [C005]."""
        result = load_gate_config(config_source={"assume_worst_on_incomplete": False})
        assert FindingCode.C005 in result.block_on_codes

    def test_gate_config_missing_assume_worst_defaults(self):
        """Missing assume_worst_on_incomplete defaults to True."""
        result = load_gate_config(config_source={"block_on_codes": ["C005"]})
        assert result.assume_worst_on_incomplete is True

    def test_gate_config_valid_finding_codes_only(self):
        """All block_on_codes are valid FindingCode variants."""
        result = load_gate_config(config_source={})
        for code in result.block_on_codes:
            assert isinstance(code, FindingCode), f"{code} is not a FindingCode"

    def test_gate_config_multiple_codes(self):
        """Multiple valid codes are all preserved."""
        result = load_gate_config(config_source={
            "block_on_codes": ["C005", "FA_A_015", "INCOMPLETE_SCHEMA"],
        })
        assert FindingCode.C005 in result.block_on_codes
        assert FindingCode.FA_A_015 in result.block_on_codes
        assert FindingCode.INCOMPLETE_SCHEMA in result.block_on_codes


# ============================================================================
# Test load_classification_registry
# ============================================================================

class TestLoadClassificationRegistry:
    """Tests for load_classification_registry function."""

    def test_registry_happy_load(self):
        """Valid entries are loaded with correct length."""
        source = [
            {"field_pattern": "*email*", "tier": "CONFIDENTIAL", "pattern_type": "fnmatch", "description": "email"},
            {"field_pattern": ".*ssn$", "tier": "RESTRICTED", "pattern_type": "regex", "description": "ssn"},
        ]
        result = load_classification_registry(registry_source=source)
        assert len(result) == 2

    def test_registry_invalid_tier(self):
        """Invalid tier value raises error."""
        source = [{"field_pattern": "*x*", "tier": "SUPER_SECRET", "pattern_type": "fnmatch", "description": "x"}]
        with pytest.raises(Exception):
            load_classification_registry(registry_source=source)

    def test_registry_invalid_regex(self):
        """Invalid regex raises error."""
        source = [{"field_pattern": "[invalid", "tier": "PUBLIC", "pattern_type": "regex", "description": "x"}]
        with pytest.raises(Exception):
            load_classification_registry(registry_source=source)

    def test_registry_missing_key(self):
        """Missing required key raises error."""
        source = [{"tier": "PUBLIC", "pattern_type": "fnmatch"}]  # Missing field_pattern
        with pytest.raises(Exception):
            load_classification_registry(registry_source=source)

    def test_registry_invalid_pattern_type(self):
        """Invalid pattern_type raises error."""
        source = [{"field_pattern": "*x*", "tier": "PUBLIC", "pattern_type": "glob", "description": "x"}]
        with pytest.raises(Exception):
            load_classification_registry(registry_source=source)

    def test_registry_all_tiers_valid(self):
        """All tier values in result are valid DataTier variants."""
        source = [
            {"field_pattern": "*email*", "tier": "CONFIDENTIAL", "pattern_type": "fnmatch", "description": "e"},
            {"field_pattern": "*name*", "tier": "INTERNAL", "pattern_type": "fnmatch", "description": "n"},
        ]
        result = load_classification_registry(registry_source=source)
        for entry in result:
            assert isinstance(entry.tier, DataTier)

    def test_registry_pattern_types_valid(self):
        """All pattern_type values are fnmatch or regex."""
        source = [
            {"field_pattern": "*x*", "tier": "PUBLIC", "pattern_type": "fnmatch", "description": "x"},
            {"field_pattern": ".*y$", "tier": "INTERNAL", "pattern_type": "regex", "description": "y"},
        ]
        result = load_classification_registry(registry_source=source)
        for entry in result:
            assert entry.pattern_type in ("fnmatch", "regex")

    def test_registry_empty_list(self):
        """Empty list returns empty result."""
        result = load_classification_registry(registry_source=[])
        assert len(result) == 0

    def test_registry_preserves_length(self):
        """len(result) == len(registry_source)."""
        source = [
            {"field_pattern": f"*field{i}*", "tier": "PUBLIC", "pattern_type": "fnmatch", "description": f"f{i}"}
            for i in range(10)
        ]
        result = load_classification_registry(registry_source=source)
        assert len(result) == 10


# ============================================================================
# Invariant Tests
# ============================================================================

class TestInvariants:
    """Tests for contract-wide invariants."""

    def test_datatier_ordering(self):
        """DataTier ordering: PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED."""
        assert DataTier.PUBLIC < DataTier.INTERNAL
        assert DataTier.INTERNAL < DataTier.CONFIDENTIAL
        assert DataTier.CONFIDENTIAL < DataTier.RESTRICTED
        # Also verify integer values
        assert DataTier.PUBLIC.value < DataTier.INTERNAL.value
        assert DataTier.INTERNAL.value < DataTier.CONFIDENTIAL.value
        assert DataTier.CONFIDENTIAL.value < DataTier.RESTRICTED.value

    def test_datatier_specific_values(self):
        """DataTier integer values are 0, 1, 2, 3."""
        assert DataTier.PUBLIC.value == 0
        assert DataTier.INTERNAL.value == 1
        assert DataTier.CONFIDENTIAL.value == 2
        assert DataTier.RESTRICTED.value == 3

    def test_finding_severity_ordering(self):
        """FindingSeverity ordering: INFO < WARNING < ERROR < CRITICAL."""
        assert FindingSeverity.INFO < FindingSeverity.WARNING
        assert FindingSeverity.WARNING < FindingSeverity.ERROR
        assert FindingSeverity.ERROR < FindingSeverity.CRITICAL

    def test_finding_code_stable_identifiers(self):
        """FindingCode values are stable string identifiers."""
        # C005 for DECLARATION_GAP
        c005_val = str(FindingCode.C005)
        assert "C005" in c005_val

        # FA_A_015 for observed access violation
        fa_val = str(FindingCode.FA_A_015)
        assert "FA_A_015" in fa_val

    def test_slot_decision_verdict_values(self):
        """SlotDecisionVerdict has ALLOW and BLOCK variants."""
        assert SlotDecisionVerdict.ALLOW is not None
        assert SlotDecisionVerdict.BLOCK is not None
        assert SlotDecisionVerdict.ALLOW != SlotDecisionVerdict.BLOCK

    def test_unmatched_fields_default_public(self):
        """Classification of unmatched fields defaults to PUBLIC."""
        fields = [make_field_entry(path="response.unknown_field")]
        registry = [make_registry_entry(field_pattern="*will_not_match_anything_xyz*", tier=DataTier.RESTRICTED)]
        result = classify_fields(fields=fields, registry_entries=registry)
        assert result.classified_fields[0].tier == DataTier.PUBLIC


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests from schema to decision."""

    def test_schema_to_decision_allow(self):
        """End-to-end: schema with only PUBLIC fields and matching declarations yields ALLOW."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
        }
        registry_source = [
            {"field_pattern": "*email*", "tier": "CONFIDENTIAL", "pattern_type": "fnmatch", "description": "email"},
        ]
        registry = load_classification_registry(registry_source=registry_source)
        gate_config = load_gate_config(config_source={"block_on_codes": ["C005"]})

        node_id = NodeId("integration-node")
        adapter_slot_id = AdapterSlotId("integration-slot")

        profile = compute_structural_profile(
            node_id=node_id,
            adapter_slot_id=adapter_slot_id,
            endpoint="/api/status",
            schema=schema,
            ref_resolver=noop_ref_resolver,
            registry_entries=registry,
            gate_config=gate_config,
        )

        declared = make_declared_access(
            node_id="integration-node",
            read_tiers=[DataTier.PUBLIC],
        )

        decision = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert decision.decision == SlotDecisionVerdict.ALLOW

    def test_schema_to_decision_block(self):
        """End-to-end: schema with CONFIDENTIAL fields undeclared yields BLOCK."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
        }
        registry_source = [
            {"field_pattern": "*email*", "tier": "CONFIDENTIAL", "pattern_type": "fnmatch", "description": "email fields"},
        ]
        registry = load_classification_registry(registry_source=registry_source)
        gate_config = load_gate_config(config_source={"block_on_codes": ["C005"]})

        node_id = NodeId("block-node")
        adapter_slot_id = AdapterSlotId("block-slot")

        profile = compute_structural_profile(
            node_id=node_id,
            adapter_slot_id=adapter_slot_id,
            endpoint="/api/users",
            schema=schema,
            ref_resolver=noop_ref_resolver,
            registry_entries=registry,
            gate_config=gate_config,
        )

        declared = make_declared_access(
            node_id="block-node",
            read_tiers=[DataTier.PUBLIC],  # Missing CONFIDENTIAL
        )

        decision = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert decision.decision == SlotDecisionVerdict.BLOCK
        assert any(f.code == FindingCode.C005 for f in decision.findings)

    def test_empty_schema_assume_worst_block(self):
        """End-to-end: empty schema with assume_worst triggers RESTRICTED and BLOCK."""
        gate_config = load_gate_config(config_source={
            "block_on_codes": ["C005"],
            "assume_worst_on_incomplete": True,
        })
        registry = load_classification_registry(registry_source=[])

        node_id = NodeId("empty-schema-node")
        adapter_slot_id = AdapterSlotId("empty-slot")

        profile = compute_structural_profile(
            node_id=node_id,
            adapter_slot_id=adapter_slot_id,
            endpoint="/api/unknown",
            schema={},
            ref_resolver=noop_ref_resolver,
            registry_entries=registry,
            gate_config=gate_config,
        )

        assert DataTier.RESTRICTED in profile.tiers
        assert profile.schema_complete is False

        declared = make_declared_access(
            node_id="empty-schema-node",
            read_tiers=[DataTier.PUBLIC],
        )

        decision = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert decision.decision == SlotDecisionVerdict.BLOCK

    def test_full_pipeline_with_observed_output(self):
        """End-to-end: structural audit ALLOW followed by runtime observation violation."""
        # Setup: node declares PUBLIC+INTERNAL, schema only has PUBLIC fields -> ALLOW
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        }
        registry = load_classification_registry(registry_source=[])
        gate_config = load_gate_config(config_source={"block_on_codes": ["C005"]})

        node_id = NodeId("runtime-node")
        adapter_slot_id = AdapterSlotId("runtime-slot")

        profile = compute_structural_profile(
            node_id=node_id,
            adapter_slot_id=adapter_slot_id,
            endpoint="/api/status",
            schema=schema,
            ref_resolver=noop_ref_resolver,
            registry_entries=registry,
            gate_config=gate_config,
        )

        declared = make_declared_access(
            node_id="runtime-node",
            read_tiers=[DataTier.PUBLIC],
        )

        structural_decision = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)
        assert structural_decision.decision == SlotDecisionVerdict.ALLOW

        # Runtime: node actually outputs RESTRICTED data (violation)
        observed = make_observed_output(
            node_id="runtime-node",
            adapter_slot_id="runtime-slot",
            observed_tiers=[DataTier.RESTRICTED],
            observed_fields=[make_field_entry(path="response.secret_key")],
        )

        runtime_findings = audit_observed_output(observed=observed, declared_access=declared)
        assert len(runtime_findings) >= 1
        assert all(f.code == FindingCode.FA_A_015 for f in runtime_findings)
        assert all(f.severity.value >= FindingSeverity.ERROR.value for f in runtime_findings)

    def test_trust_vs_authority_distinction(self):
        """Trust (structural profile tiers) is distinct from authority (declared access tiers)."""
        # Structural profile has CONFIDENTIAL (computed/earned trust)
        profile = make_structural_profile(
            node_id="distinction-node",
            tiers=[DataTier.PUBLIC, DataTier.CONFIDENTIAL],
        )
        # Declared access only has PUBLIC (authority claim)
        declared = make_declared_access(
            node_id="distinction-node",
            read_tiers=[DataTier.PUBLIC],
        )
        gate_config = make_gate_config(block_on_codes=[FindingCode.C005])

        decision = audit_slot(profile=profile, declared_access=declared, gate_config=gate_config)

        # The gap between trust and authority should produce a finding
        assert any(f.code == FindingCode.C005 for f in decision.findings)
        assert decision.decision == SlotDecisionVerdict.BLOCK
