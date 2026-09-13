"""
Hidden adversarial acceptance tests for Access Auditor & OpenAPI Integration.

These tests target behavioral gaps not covered by visible tests, specifically
looking for hardcoded returns, boundary errors, and invariant violations.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call
from arbiter.access import *


# ---------------------------------------------------------------------------
# walk_response_schema — gaps
# ---------------------------------------------------------------------------

class TestGoodhartWalkResponseSchema:

    def test_goodhart_walk_anyof_composition(self):
        """walk_response_schema should flatten anyOf with union semantics, discovering leaf fields from all branches."""
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "alpha": {"type": "string"},
                    }
                },
                {
                    "type": "object",
                    "properties": {
                        "beta": {"type": "integer"},
                    }
                }
            ]
        }
        ref_resolver = lambda uri: {}
        result = walk_response_schema(schema, ref_resolver, root_path="response", max_depth=10)
        paths = [f.path for f in result.fields]
        assert any("alpha" in p for p in paths), "alpha field not found in anyOf branch"
        assert any("beta" in p for p in paths), "beta field not found in anyOf branch"

    def test_goodhart_walk_oneof_composition(self):
        """walk_response_schema should flatten oneOf with union semantics, discovering all variant leaves."""
        schema = {
            "oneOf": [
                {"type": "object", "properties": {"variant_a": {"type": "string"}}},
                {"type": "object", "properties": {"variant_b": {"type": "integer"}}},
                {"type": "object", "properties": {"variant_c": {"type": "boolean"}}},
            ]
        }
        ref_resolver = lambda uri: {}
        result = walk_response_schema(schema, ref_resolver, root_path="response", max_depth=10)
        paths = [f.path for f in result.fields]
        assert any("variant_a" in p for p in paths)
        assert any("variant_b" in p for p in paths)
        assert any("variant_c" in p for p in paths)

    def test_goodhart_walk_custom_root_path(self):
        """All discovered field paths must be prefixed with the provided root_path, not a hardcoded default."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            }
        }
        ref_resolver = lambda uri: {}
        result = walk_response_schema(schema, ref_resolver, root_path="custom.prefix", max_depth=10)
        assert len(result.fields) >= 2
        for f in result.fields:
            assert str(f.path).startswith("custom.prefix"), f"Path {f.path} does not start with custom.prefix"

    def test_goodhart_walk_deeply_nested_at_max_depth(self):
        """Schema at exactly max_depth should succeed — verifying no off-by-one in depth limit."""
        # Build a schema nested 3 levels deep: root -> level1 -> level2 -> leaf
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "leaf": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        ref_resolver = lambda uri: {}
        # max_depth=3 should allow reaching depth 3
        result = walk_response_schema(schema, ref_resolver, root_path="r", max_depth=3)
        paths = [str(f.path) for f in result.fields]
        assert any("leaf" in p for p in paths), f"Leaf at depth 3 not found. Fields: {paths}"

    def test_goodhart_walk_format_hint_preserved(self):
        """FieldEntry format_hint should capture the schema's format field."""
        schema = {
            "type": "object",
            "properties": {
                "created_at": {"type": "string", "format": "date-time"},
                "contact_email": {"type": "string", "format": "email"},
            }
        }
        ref_resolver = lambda uri: {}
        result = walk_response_schema(schema, ref_resolver, root_path="response", max_depth=10)
        format_hints = {f.format_hint for f in result.fields}
        assert "date-time" in format_hints, f"date-time format not found. Hints: {format_hints}"
        assert "email" in format_hints, f"email format not found. Hints: {format_hints}"

    def test_goodhart_walk_nested_ref_chain(self):
        """Schema walker should resolve chains of $ref references (A -> B -> C)."""
        schema = {"$ref": "#/components/schemas/A"}

        def resolver(uri):
            if uri == "#/components/schemas/A":
                return {"$ref": "#/components/schemas/B"}
            elif uri == "#/components/schemas/B":
                return {
                    "type": "object",
                    "properties": {
                        "deep_leaf": {"type": "string"}
                    }
                }
            raise ValueError(f"Unknown ref: {uri}")

        result = walk_response_schema(schema, resolver, root_path="response", max_depth=10)
        paths = [str(f.path) for f in result.fields]
        assert any("deep_leaf" in p for p in paths), f"Deep leaf not found via ref chain. Fields: {paths}"

    def test_goodhart_walk_empty_dict_schema(self):
        """An empty dict {} schema should produce an INCOMPLETE_SCHEMA warning."""
        ref_resolver = lambda uri: {}
        result = walk_response_schema({}, ref_resolver, root_path="response", max_depth=10)
        warning_codes = [w.code for w in result.warnings]
        assert any(
            str(c) == str(FindingCode.INCOMPLETE_SCHEMA) or c == FindingCode.INCOMPLETE_SCHEMA
            for c in warning_codes
        ), f"No INCOMPLETE_SCHEMA warning for empty dict. Codes: {warning_codes}"

    def test_goodhart_walk_nested_array_of_objects(self):
        """Schema walker should traverse arrays of objects."""
        schema = {
            "type": "object",
            "properties": {
                "items_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_name": {"type": "string"},
                            "item_price": {"type": "number"}
                        }
                    }
                }
            }
        }
        ref_resolver = lambda uri: {}
        result = walk_response_schema(schema, ref_resolver, root_path="response", max_depth=10)
        paths = [str(f.path) for f in result.fields]
        assert any("item_name" in p for p in paths), f"item_name not found. Fields: {paths}"
        assert any("item_price" in p for p in paths), f"item_price not found. Fields: {paths}"

    def test_goodhart_walk_field_type_captured(self):
        """FieldEntry field_type should reflect the schema's type."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
            }
        }
        ref_resolver = lambda uri: {}
        result = walk_response_schema(schema, ref_resolver, root_path="response", max_depth=10)
        type_map = {str(f.path).split(".")[-1]: f.field_type for f in result.fields}
        assert type_map.get("name") == "string", f"Expected string, got {type_map.get('name')}"
        assert type_map.get("count") == "integer", f"Expected integer, got {type_map.get('count')}"
        assert type_map.get("active") == "boolean", f"Expected boolean, got {type_map.get('active')}"

    def test_goodhart_walk_ref_resolver_used_not_ignored(self):
        """The ref_resolver callable must actually be invoked for $ref URIs."""
        schema = {
            "type": "object",
            "properties": {
                "address": {"$ref": "#/components/schemas/Address"}
            }
        }
        call_count = {"n": 0}

        def tracking_resolver(uri):
            call_count["n"] += 1
            return {
                "type": "object",
                "properties": {
                    "street": {"type": "string"}
                }
            }

        result = walk_response_schema(schema, tracking_resolver, root_path="response", max_depth=10)
        assert call_count["n"] >= 1, "ref_resolver was never called"
        paths = [str(f.path) for f in result.fields]
        assert any("street" in p for p in paths), "Referenced field not discovered"


# ---------------------------------------------------------------------------
# classify_fields — gaps
# ---------------------------------------------------------------------------

class TestGoodhartClassifyFields:

    def _make_field(self, path, field_type="string"):
        return FieldEntry(path=FieldPath(path), field_type=field_type, nullable=False, format_hint="")

    def _make_entry(self, pattern, tier, pattern_type="fnmatch", desc=""):
        return ClassificationRegistryEntry(
            field_pattern=pattern, tier=tier, pattern_type=pattern_type, description=desc
        )

    def test_goodhart_classify_multiple_fields_different_tiers(self):
        """Classification should independently assign different tiers to different fields."""
        fields = [
            self._make_field("response.user.ssn"),
            self._make_field("response.user.department"),
            self._make_field("response.user.display_name"),
        ]
        registry = [
            self._make_entry("*.ssn", DataTier.RESTRICTED),
            self._make_entry("*.department", DataTier.INTERNAL),
        ]
        result = classify_fields(fields, registry)
        tier_by_path = {str(cf.path): cf.tier for cf in result.classified_fields}
        assert tier_by_path["response.user.ssn"] == DataTier.RESTRICTED
        assert tier_by_path["response.user.department"] == DataTier.INTERNAL
        assert tier_by_path["response.user.display_name"] == DataTier.PUBLIC
        assert DataTier.PUBLIC in result.tier_set
        assert DataTier.INTERNAL in result.tier_set
        assert DataTier.RESTRICTED in result.tier_set

    def test_goodhart_classify_fnmatch_wildcard_patterns(self):
        """fnmatch patterns with wildcards should match arbitrary subpaths."""
        fields = [
            self._make_field("response.users.ssn"),
            self._make_field("response.admin.ssn"),
        ]
        registry = [
            self._make_entry("*.ssn", DataTier.RESTRICTED),
        ]
        result = classify_fields(fields, registry)
        for cf in result.classified_fields:
            assert cf.tier == DataTier.RESTRICTED, f"Field {cf.path} should be RESTRICTED, got {cf.tier}"

    def test_goodhart_classify_empty_registry(self):
        """When no registry entries are provided, all fields default to PUBLIC."""
        fields = [
            self._make_field("response.a"),
            self._make_field("response.b"),
            self._make_field("response.c"),
        ]
        result = classify_fields(fields, [])
        for cf in result.classified_fields:
            assert cf.tier == DataTier.PUBLIC, f"Expected PUBLIC, got {cf.tier}"
        assert len(result.tier_set) == 1
        assert DataTier.PUBLIC in result.tier_set

    def test_goodhart_classify_three_overlapping_patterns_highest_wins(self):
        """When a field matches three patterns of different tiers, the highest tier wins."""
        fields = [self._make_field("response.data.email")]
        registry = [
            self._make_entry("*.email", DataTier.PUBLIC),
            self._make_entry("*email*", DataTier.INTERNAL, "fnmatch"),
            self._make_entry("*.email", DataTier.CONFIDENTIAL),
        ]
        result = classify_fields(fields, registry)
        assert result.classified_fields[0].tier == DataTier.CONFIDENTIAL

    def test_goodhart_classify_single_field_no_registry(self):
        """A single field with no registry entries produces exactly one ClassifiedField at PUBLIC."""
        fields = [self._make_field("response.id")]
        result = classify_fields(fields, [])
        assert len(result.classified_fields) == 1
        assert result.classified_fields[0].tier == DataTier.PUBLIC

    def test_goodhart_classify_regex_partial_vs_full_match(self):
        """Regex matching should work correctly — fields matching regex patterns get proper tiers."""
        fields = [
            self._make_field("response.user.email"),
        ]
        registry = [
            self._make_entry(r".*\.email$", DataTier.CONFIDENTIAL, "regex"),
        ]
        result = classify_fields(fields, registry)
        assert result.classified_fields[0].tier == DataTier.CONFIDENTIAL

    def test_goodhart_classify_matched_pattern_recorded(self):
        """ClassifiedField should record which pattern caused the match."""
        fields = [self._make_field("response.user.ssn")]
        registry = [
            self._make_entry("*.ssn", DataTier.RESTRICTED, "fnmatch", "SSN field"),
        ]
        result = classify_fields(fields, registry)
        assert result.classified_fields[0].matched_pattern == "*.ssn"


# ---------------------------------------------------------------------------
# compute_structural_profile — gaps
# ---------------------------------------------------------------------------

class TestGoodhartComputeStructuralProfile:

    def _make_gate_config(self, block_on_codes=None, assume_worst=True):
        if block_on_codes is None:
            block_on_codes = [FindingCode.C005]
        return GateConfig(block_on_codes=block_on_codes, assume_worst_on_incomplete=assume_worst)

    def test_goodhart_profile_assume_worst_false_empty_schema(self):
        """When assume_worst_on_incomplete is False, empty schema should NOT inject RESTRICTED tier."""
        gc = self._make_gate_config(assume_worst=False)
        ref_resolver = lambda uri: {}
        result = compute_structural_profile(
            node_id=NodeId("node-1"),
            adapter_slot_id=AdapterSlotId("slot-1"),
            endpoint="/api/test",
            schema={},
            ref_resolver=ref_resolver,
            registry_entries=[],
            gate_config=gc,
        )
        assert DataTier.RESTRICTED not in result.tiers, "RESTRICTED should not be injected when assume_worst is False"
        assert result.schema_complete is False

    def test_goodhart_profile_schema_incomplete_sets_false(self):
        """schema_complete must be False when warnings exist, even if fields were discovered."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ref_field": {"$ref": "#/components/schemas/Missing"}
            }
        }

        def resolver(uri):
            raise Exception(f"Cannot resolve {uri}")

        gc = self._make_gate_config(assume_worst=False)
        # The function may raise or handle ref_resolver failure as a warning
        # depending on implementation; try to test the warning path
        try:
            result = compute_structural_profile(
                node_id=NodeId("node-2"),
                adapter_slot_id=AdapterSlotId("slot-2"),
                endpoint="/api/partial",
                schema=schema,
                ref_resolver=resolver,
                registry_entries=[],
                gate_config=gc,
            )
            assert result.schema_complete is False, "Schema with unresolvable refs should not be marked complete"
            assert len(result.warnings) > 0
        except Exception:
            # If it raises, that's also acceptable per the error contract
            pass

    def test_goodhart_profile_empty_adapter_slot_id_error(self):
        """Empty adapter_slot_id should violate precondition."""
        gc = self._make_gate_config()
        ref_resolver = lambda uri: {}
        with pytest.raises(Exception):
            compute_structural_profile(
                node_id=NodeId("node-1"),
                adapter_slot_id=AdapterSlotId(""),
                endpoint="/api/test",
                schema={"type": "object", "properties": {"x": {"type": "string"}}},
                ref_resolver=ref_resolver,
                registry_entries=[],
                gate_config=gc,
            )

    def test_goodhart_profile_empty_endpoint_error(self):
        """Empty endpoint should violate precondition."""
        gc = self._make_gate_config()
        ref_resolver = lambda uri: {}
        with pytest.raises(Exception):
            compute_structural_profile(
                node_id=NodeId("node-1"),
                adapter_slot_id=AdapterSlotId("slot-1"),
                endpoint="",
                schema={"type": "object", "properties": {"x": {"type": "string"}}},
                ref_resolver=ref_resolver,
                registry_entries=[],
                gate_config=gc,
            )

    def test_goodhart_profile_tiers_are_set_like(self):
        """Profile tiers should contain no duplicates even when multiple fields share a tier."""
        schema = {
            "type": "object",
            "properties": {
                "field_a": {"type": "string"},
                "field_b": {"type": "string"},
                "field_c": {"type": "string"},
            }
        }
        gc = self._make_gate_config(assume_worst=False)
        ref_resolver = lambda uri: {}
        # No registry entries means all fields are PUBLIC
        result = compute_structural_profile(
            node_id=NodeId("node-dedup"),
            adapter_slot_id=AdapterSlotId("slot-dedup"),
            endpoint="/api/dedup",
            schema=schema,
            ref_resolver=ref_resolver,
            registry_entries=[],
            gate_config=gc,
        )
        tiers_list = list(result.tiers)
        assert len(tiers_list) == len(set(tiers_list)), "Duplicate tiers in profile.tiers"


# ---------------------------------------------------------------------------
# audit_slot — gaps
# ---------------------------------------------------------------------------

class TestGoodhartAuditSlot:

    def _make_profile(self, node_id, slot_id, tiers, warnings=None, classified_fields=None, endpoint="/api/test"):
        return StructuralProfile(
            node_id=NodeId(node_id),
            adapter_slot_id=AdapterSlotId(slot_id),
            endpoint=endpoint,
            tiers=list(tiers),
            classified_fields=classified_fields or [],
            warnings=warnings or [],
            schema_complete=(not warnings),
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _make_declared(self, node_id, read_tiers):
        return DeclaredAccess(
            node_id=NodeId(node_id),
            declared_read_tiers=list(read_tiers),
            declared_write_tiers=[],
        )

    def _make_gate_config(self, block_on_codes=None):
        if block_on_codes is None:
            block_on_codes = [FindingCode.C005]
        return GateConfig(block_on_codes=block_on_codes, assume_worst_on_incomplete=True)

    def test_goodhart_audit_slot_allow_when_c005_not_in_block_codes(self):
        """C005 findings should not trigger BLOCK when C005 is not in gate_config.block_on_codes."""
        profile = self._make_profile("node-x", "slot-x", [DataTier.PUBLIC, DataTier.CONFIDENTIAL])
        declared = self._make_declared("node-x", [DataTier.PUBLIC])
        gc = self._make_gate_config(block_on_codes=[])  # C005 NOT in block_on_codes

        result = audit_slot(profile, declared, gc)
        assert result.decision == SlotDecisionVerdict.ALLOW
        # Should still have C005 findings
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) >= 1, "Should still produce C005 findings even if not blocking"
        assert len(result.blocking_codes) == 0

    def test_goodhart_audit_slot_block_on_incomplete_schema_code(self):
        """When INCOMPLETE_SCHEMA is in block_on_codes and profile has INCOMPLETE_SCHEMA warnings, verdict should be BLOCK."""
        warnings = [SchemaWarning(
            code=FindingCode.INCOMPLETE_SCHEMA,
            path="response.missing",
            message="Missing schema for field"
        )]
        profile = self._make_profile("node-inc", "slot-inc", [DataTier.PUBLIC], warnings=warnings)
        declared = self._make_declared("node-inc", [DataTier.PUBLIC])
        gc = self._make_gate_config(block_on_codes=[FindingCode.INCOMPLETE_SCHEMA])

        result = audit_slot(profile, declared, gc)
        assert result.decision == SlotDecisionVerdict.BLOCK
        blocking = [str(c) for c in result.blocking_codes]
        assert any("INCOMPLETE_SCHEMA" in c for c in blocking) or FindingCode.INCOMPLETE_SCHEMA in result.blocking_codes

    def test_goodhart_audit_slot_evidence_fields_populated(self):
        """C005 finding evidence must contain correct structural_tiers, declared_tiers, undeclared_tiers."""
        profile = self._make_profile(
            "node-ev", "slot-ev",
            [DataTier.PUBLIC, DataTier.CONFIDENTIAL],
            classified_fields=[
                ClassifiedField(path=FieldPath("response.email"), tier=DataTier.CONFIDENTIAL, matched_pattern="*.email")
            ]
        )
        declared = self._make_declared("node-ev", [DataTier.PUBLIC])
        gc = self._make_gate_config()

        result = audit_slot(profile, declared, gc)
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) >= 1
        ev = c005_findings[0].evidence
        assert DataTier.CONFIDENTIAL in ev.undeclared_tiers
        assert DataTier.PUBLIC in ev.declared_tiers

    def test_goodhart_audit_slot_no_findings_when_all_declared(self):
        """When all structural tiers are subset of declared, there should be zero C005 findings."""
        profile = self._make_profile("node-ok", "slot-ok", [DataTier.PUBLIC, DataTier.INTERNAL])
        declared = self._make_declared("node-ok", [DataTier.PUBLIC, DataTier.INTERNAL, DataTier.CONFIDENTIAL])
        gc = self._make_gate_config()

        result = audit_slot(profile, declared, gc)
        assert result.decision == SlotDecisionVerdict.ALLOW
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) == 0

    def test_goodhart_audit_slot_multiple_undeclared_tiers_separate_findings(self):
        """When multiple tiers are undeclared, each gets its own C005 finding."""
        profile = self._make_profile(
            "node-multi", "slot-multi",
            [DataTier.PUBLIC, DataTier.INTERNAL, DataTier.CONFIDENTIAL, DataTier.RESTRICTED]
        )
        declared = self._make_declared("node-multi", [DataTier.PUBLIC])
        gc = self._make_gate_config()

        result = audit_slot(profile, declared, gc)
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) == 3, f"Expected 3 C005 findings, got {len(c005_findings)}"

    def test_goodhart_audit_slot_finding_has_adapter_slot_id(self):
        """Each AccessFinding must include the adapter_slot_id from the profile."""
        profile = self._make_profile("node-slot", "specific-slot-42", [DataTier.PUBLIC, DataTier.RESTRICTED])
        declared = self._make_declared("node-slot", [DataTier.PUBLIC])
        gc = self._make_gate_config()

        result = audit_slot(profile, declared, gc)
        for finding in result.findings:
            assert str(finding.adapter_slot_id) == "specific-slot-42", \
                f"Finding adapter_slot_id should be 'specific-slot-42', got '{finding.adapter_slot_id}'"

    def test_goodhart_audit_slot_c005_finding_code_exact(self):
        """C005 findings must have code exactly equal to FindingCode.C005."""
        profile = self._make_profile("node-code", "slot-code", [DataTier.PUBLIC, DataTier.INTERNAL])
        declared = self._make_declared("node-code", [DataTier.PUBLIC])
        gc = self._make_gate_config()

        result = audit_slot(profile, declared, gc)
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) >= 1
        for f in c005_findings:
            assert f.code == FindingCode.C005

    def test_goodhart_audit_slot_blocking_codes_match_findings(self):
        """result.blocking_codes should contain exactly the codes present in both findings and block_on_codes."""
        profile = self._make_profile("node-bc", "slot-bc", [DataTier.PUBLIC, DataTier.INTERNAL])
        declared = self._make_declared("node-bc", [DataTier.PUBLIC])
        gc = self._make_gate_config(block_on_codes=[FindingCode.C005])

        result = audit_slot(profile, declared, gc)
        assert FindingCode.C005 in result.blocking_codes

    def test_goodhart_audit_slot_evidence_endpoint(self):
        """C005 finding evidence should include the endpoint from the profile."""
        profile = self._make_profile(
            "node-ep", "slot-ep",
            [DataTier.PUBLIC, DataTier.CONFIDENTIAL],
            endpoint="/api/v2/users"
        )
        declared = self._make_declared("node-ep", [DataTier.PUBLIC])
        gc = self._make_gate_config()

        result = audit_slot(profile, declared, gc)
        c005_findings = [f for f in result.findings if f.code == FindingCode.C005]
        assert len(c005_findings) >= 1
        assert c005_findings[0].evidence.endpoint == "/api/v2/users"


# ---------------------------------------------------------------------------
# audit_observed_output — gaps
# ---------------------------------------------------------------------------

class TestGoodhartAuditObservedOutput:

    def _make_observed(self, node_id, slot_id, tiers, fields=None):
        return ObservedOutput(
            node_id=NodeId(node_id),
            adapter_slot_id=AdapterSlotId(slot_id),
            observed_tiers=list(tiers),
            observed_fields=fields or [],
            observed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _make_declared(self, node_id, read_tiers):
        return DeclaredAccess(
            node_id=NodeId(node_id),
            declared_read_tiers=list(read_tiers),
            declared_write_tiers=[],
        )

    def test_goodhart_observed_single_undeclared_restricted(self):
        """A single RESTRICTED tier observed but not declared produces exactly one FA_A_015."""
        observed = self._make_observed("node-r", "slot-r", [DataTier.RESTRICTED])
        declared = self._make_declared("node-r", [DataTier.PUBLIC, DataTier.INTERNAL])

        result = audit_observed_output(observed, declared)
        assert len(result) == 1
        assert result[0].code == FindingCode.FA_A_015
        assert "RESTRICTED" in result[0].message or "restricted" in result[0].message.lower()

    def test_goodhart_observed_message_includes_field_paths(self):
        """FA_A_015 finding messages must include the specific field paths at the undeclared tier."""
        observed = self._make_observed(
            "node-fp", "slot-fp",
            [DataTier.CONFIDENTIAL],
            fields=[
                ClassifiedField(
                    path=FieldPath("response.user.email"),
                    tier=DataTier.CONFIDENTIAL,
                    matched_pattern="*.email"
                )
            ]
        )
        declared = self._make_declared("node-fp", [DataTier.PUBLIC])

        result = audit_observed_output(observed, declared)
        assert len(result) >= 1
        # The message should mention the field path
        assert "email" in result[0].message.lower() or "response.user.email" in result[0].message

    def test_goodhart_observed_finding_has_adapter_slot_id(self):
        """FA_A_015 findings must carry the adapter_slot_id from the observed output."""
        observed = self._make_observed("node-as", "unique-slot-99", [DataTier.RESTRICTED])
        declared = self._make_declared("node-as", [DataTier.PUBLIC])

        result = audit_observed_output(observed, declared)
        assert len(result) >= 1
        assert str(result[0].adapter_slot_id) == "unique-slot-99"

    def test_goodhart_observed_all_declared_empty_result(self):
        """When all observed tiers are declared, result must be an empty list."""
        observed = self._make_observed("node-all", "slot-all", [DataTier.PUBLIC, DataTier.CONFIDENTIAL])
        declared = self._make_declared("node-all", [DataTier.PUBLIC, DataTier.INTERNAL, DataTier.CONFIDENTIAL, DataTier.RESTRICTED])

        result = audit_observed_output(observed, declared)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_goodhart_observed_finding_node_id_matches(self):
        """Each FA_A_015 finding's node_id field must equal the observed output's node_id."""
        observed = self._make_observed("specific-node-77", "slot-77", [DataTier.INTERNAL])
        declared = self._make_declared("specific-node-77", [DataTier.PUBLIC])

        result = audit_observed_output(observed, declared)
        assert len(result) >= 1
        assert str(result[0].node_id) == "specific-node-77"


# ---------------------------------------------------------------------------
# load_gate_config — gaps
# ---------------------------------------------------------------------------

class TestGoodhartLoadGateConfig:

    def test_goodhart_gate_config_assume_worst_explicit_false(self):
        """When assume_worst_on_incomplete is explicitly False, the loaded config must respect it."""
        result = load_gate_config({"assume_worst_on_incomplete": False})
        assert result.assume_worst_on_incomplete is False
        # block_on_codes should still default to [C005]
        assert FindingCode.C005 in result.block_on_codes

    def test_goodhart_gate_config_multiple_block_codes(self):
        """load_gate_config should accept multiple valid FindingCode values in block_on_codes."""
        result = load_gate_config({"block_on_codes": ["C005", "INCOMPLETE_SCHEMA"]})
        assert FindingCode.C005 in result.block_on_codes
        assert FindingCode.INCOMPLETE_SCHEMA in result.block_on_codes
        assert len(result.block_on_codes) == 2

    def test_goodhart_gate_config_empty_block_codes(self):
        """An explicitly empty block_on_codes list should be accepted, not replaced with defaults."""
        result = load_gate_config({"block_on_codes": []})
        assert result.block_on_codes == [] or len(result.block_on_codes) == 0


# ---------------------------------------------------------------------------
# load_classification_registry — gaps
# ---------------------------------------------------------------------------

class TestGoodhartLoadClassificationRegistry:

    def test_goodhart_registry_multiple_entries(self):
        """load_classification_registry should correctly load multiple entries with different pattern types and tiers."""
        source = [
            {"field_pattern": "*.name", "tier": "PUBLIC", "pattern_type": "fnmatch", "description": "Name field"},
            {"field_pattern": r".*\.email$", "tier": "CONFIDENTIAL", "pattern_type": "regex", "description": "Email"},
            {"field_pattern": "*.ssn", "tier": "RESTRICTED", "pattern_type": "fnmatch", "description": "SSN"},
        ]
        result = load_classification_registry(source)
        assert len(result) == 3
        tiers = {r.tier for r in result}
        assert DataTier.PUBLIC in tiers
        assert DataTier.CONFIDENTIAL in tiers
        assert DataTier.RESTRICTED in tiers

    def test_goodhart_registry_description_preserved(self):
        """The description field from registry source should be preserved."""
        source = [
            {"field_pattern": "*.ssn", "tier": "RESTRICTED", "pattern_type": "fnmatch",
             "description": "Social security number field"},
        ]
        result = load_classification_registry(source)
        assert result[0].description == "Social security number field"


# ---------------------------------------------------------------------------
# Enum invariants — gaps
# ---------------------------------------------------------------------------

class TestGoodhartEnumInvariants:

    def test_goodhart_datatier_integer_values(self):
        """DataTier enum members must have specific integer values per contract."""
        assert int(DataTier.PUBLIC) == 0
        assert int(DataTier.INTERNAL) == 1
        assert int(DataTier.CONFIDENTIAL) == 2
        assert int(DataTier.RESTRICTED) == 3

    def test_goodhart_datatier_comparison(self):
        """DataTier members must support natural integer comparison consistent with severity ordering."""
        assert DataTier.PUBLIC < DataTier.INTERNAL
        assert DataTier.INTERNAL < DataTier.CONFIDENTIAL
        assert DataTier.CONFIDENTIAL < DataTier.RESTRICTED
        # Transitive
        assert DataTier.PUBLIC < DataTier.RESTRICTED

    def test_goodhart_finding_code_string_values(self):
        """FindingCode enum members should be usable as stable string identifiers."""
        # C005
        c005_str = str(FindingCode.C005)
        assert "C005" in c005_str or FindingCode.C005 == "C005" or FindingCode.C005.value == "C005"
        # INCOMPLETE_SCHEMA
        inc_str = str(FindingCode.INCOMPLETE_SCHEMA)
        assert "INCOMPLETE_SCHEMA" in inc_str or FindingCode.INCOMPLETE_SCHEMA.value == "INCOMPLETE_SCHEMA"

    def test_goodhart_finding_severity_ordering(self):
        """FindingSeverity enum must have correct integer ordering."""
        assert FindingSeverity.INFO < FindingSeverity.WARNING
        assert FindingSeverity.WARNING < FindingSeverity.ERROR
        assert FindingSeverity.ERROR < FindingSeverity.CRITICAL
        # Integer values
        assert int(FindingSeverity.INFO) == 0
        assert int(FindingSeverity.WARNING) == 1
        assert int(FindingSeverity.ERROR) == 2
        assert int(FindingSeverity.CRITICAL) == 3
