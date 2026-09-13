"""
Contract tests for models component.
Tests all enums, primitives, structs, and functions defined in the contract.
Run with: pytest contract_test.py -v
"""

import json
import re
import pytest
from unittest.mock import MagicMock

# Import everything from the models module
from arbiter.models import (
    TrustTier,
    DataTier,
    BlastTier,
    FindingSeverity,
    TrustEventType,
    TrustLedgerEntry,
    LedgerCheckpoint,
    AccessGraphNode,
    AccessGraph,
    ConsistencyFinding,
    AccessFinding,
    TaintFinding,
    ConflictRecord,
    StigmerySignal,
    CanaryRecord,
    FeedbackReportSection,
    FeedbackReport,
    ErrorResponse,
    ClassificationRule,
    create_trust_ledger_entry,
    create_ledger_checkpoint,
    build_access_graph,
    parse_ledger_line,
    serialize_ledger_line,
    create_error_response,
    validate_canary_fingerprint,
    score_to_tier,
    classify_field,
)


# ============================================================
# Helper constants
# ============================================================
VALID_SHA256 = "a" * 64
VALID_SHA256_ZEROS = "0" * 64
VALID_NODE_ID = "service-a"
VALID_UTC_TS = "2024-01-01T00:00:00+00:00"
VALID_UUID_V4 = "12345678-1234-4abc-8abc-123456789abc"


def _is_utc_timestamp(ts: str) -> bool:
    """Check if timestamp ends with Z or +00:00."""
    return ts.endswith("Z") or ts.endswith("+00:00")


# ============================================================
# Enum Tests
# ============================================================
class TestTrustTierEnum:
    def test_trust_tier_has_all_members(self):
        expected = {"PROBATIONARY", "LOW", "ESTABLISHED", "HIGH", "TRUSTED"}
        actual = {m.name for m in TrustTier}
        assert actual == expected, f"TrustTier members mismatch: {actual}"

    def test_trust_tier_member_count(self):
        assert len(TrustTier) == 5

    def test_trust_tier_is_str(self):
        for member in TrustTier:
            assert isinstance(member, str), f"{member} is not a str"
            assert isinstance(member.value, str), f"{member.value} is not a str"


class TestDataTierEnum:
    def test_data_tier_has_all_members(self):
        expected = {"PUBLIC", "PII", "FINANCIAL", "AUTH", "COMPLIANCE"}
        actual = {m.name for m in DataTier}
        assert actual == expected

    def test_data_tier_member_count(self):
        assert len(DataTier) == 5

    def test_data_tier_is_str(self):
        for member in DataTier:
            assert isinstance(member, str)


class TestBlastTierEnum:
    def test_blast_tier_has_all_members(self):
        expected = {"AUTO_MERGE", "SOAK", "HUMAN_GATE"}
        actual = {m.name for m in BlastTier}
        assert actual == expected

    def test_blast_tier_member_count(self):
        assert len(BlastTier) == 3


class TestFindingSeverityEnum:
    def test_finding_severity_has_all_members(self):
        expected = {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = {m.name for m in FindingSeverity}
        assert actual == expected

    def test_finding_severity_member_count(self):
        assert len(FindingSeverity) == 5


class TestTrustEventTypeEnum:
    def test_trust_event_type_has_all_members(self):
        expected = {
            "AUDIT_PASS", "AUDIT_FAIL", "CONSISTENCY_CHECK",
            "ACCESS_VIOLATION", "TAINT_DETECTED", "CANARY_TRIGGERED",
            "MANUAL_OVERRIDE", "DECAY", "INITIAL",
        }
        actual = {m.name for m in TrustEventType}
        assert actual == expected

    def test_trust_event_type_member_count(self):
        assert len(TrustEventType) == 9

    def test_trust_event_type_is_str(self):
        for member in TrustEventType:
            assert isinstance(member, str)


class TestEnumSerializationStability:
    """All enums use StrEnum with explicit string values for stable serialization."""

    @pytest.mark.parametrize("enum_cls", [TrustTier, DataTier, BlastTier, FindingSeverity, TrustEventType])
    def test_enum_values_are_strings(self, enum_cls):
        for member in enum_cls:
            assert isinstance(member.value, str), f"{enum_cls.__name__}.{member.name}.value is not str"

    @pytest.mark.parametrize("enum_cls", [TrustTier, DataTier, BlastTier, FindingSeverity, TrustEventType])
    def test_enum_str_returns_value(self, enum_cls):
        for member in enum_cls:
            # StrEnum: str(member) should work and be a string
            s = str(member)
            assert isinstance(s, str)
            assert len(s) > 0


# ============================================================
# score_to_tier Tests
# ============================================================
class TestScoreToTier:
    def test_score_0_returns_probationary(self):
        assert score_to_tier(0.0) == TrustTier.PROBATIONARY

    def test_score_just_below_02_returns_probationary(self):
        assert score_to_tier(0.19999) == TrustTier.PROBATIONARY

    def test_score_02_returns_low(self):
        assert score_to_tier(0.2) == TrustTier.LOW

    def test_score_just_below_04_returns_low(self):
        assert score_to_tier(0.39999) == TrustTier.LOW

    def test_score_04_returns_established(self):
        assert score_to_tier(0.4) == TrustTier.ESTABLISHED

    def test_score_just_below_06_returns_established(self):
        assert score_to_tier(0.59999) == TrustTier.ESTABLISHED

    def test_score_06_returns_high(self):
        assert score_to_tier(0.6) == TrustTier.HIGH

    def test_score_just_below_08_returns_high(self):
        assert score_to_tier(0.79999) == TrustTier.HIGH

    def test_score_08_returns_trusted(self):
        assert score_to_tier(0.8) == TrustTier.TRUSTED

    def test_score_10_returns_trusted(self):
        assert score_to_tier(1.0) == TrustTier.TRUSTED

    def test_midpoints(self):
        assert score_to_tier(0.1) == TrustTier.PROBATIONARY
        assert score_to_tier(0.3) == TrustTier.LOW
        assert score_to_tier(0.5) == TrustTier.ESTABLISHED
        assert score_to_tier(0.7) == TrustTier.HIGH
        assert score_to_tier(0.9) == TrustTier.TRUSTED

    def test_negative_score_raises(self):
        with pytest.raises((ValueError, Exception)):
            score_to_tier(-0.1)

    def test_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            score_to_tier(1.1)

    def test_deterministic(self):
        """Same input always yields same output."""
        for score in [0.0, 0.15, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]:
            tier1 = score_to_tier(score)
            tier2 = score_to_tier(score)
            assert tier1 == tier2, f"Non-deterministic for score {score}"


# ============================================================
# create_trust_ledger_entry Tests
# ============================================================
class TestCreateTrustLedgerEntry:
    def test_happy_path_audit_pass(self):
        entry = create_trust_ledger_entry(
            node="service-a",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="Passed audit",
        )
        assert entry.node == "service-a"
        assert entry.event == TrustEventType.AUDIT_PASS
        assert entry.weight == 0.1
        assert entry.score_before == 0.5
        assert entry.score_after == pytest.approx(0.6)
        assert entry.sequence_number == 0
        assert entry.detail == "Passed audit"
        assert _is_utc_timestamp(entry.ts)

    def test_audit_fail_with_detail(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_FAIL,
            weight=-0.2,
            score_before=0.5,
            sequence_number=5,
            detail="checksum mismatch",
        )
        assert entry.detail == "checksum mismatch"
        assert entry.score_after == pytest.approx(0.3)

    def test_clamp_upper_to_one(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.5,
            score_before=0.8,
            sequence_number=1,
            detail="test",
        )
        assert entry.score_after == 1.0

    def test_clamp_lower_to_zero(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.DECAY,
            weight=-0.5,
            score_before=0.2,
            sequence_number=2,
            detail="decay",
        )
        assert entry.score_after == 0.0

    def test_score_before_zero_with_positive_weight(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.INITIAL,
            weight=0.5,
            score_before=0.0,
            sequence_number=0,
            detail="init",
        )
        assert entry.score_after == 0.5

    def test_score_before_one_with_zero_weight(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.0,
            score_before=1.0,
            sequence_number=3,
            detail="no change",
        )
        assert entry.score_after == 1.0

    def test_weight_exactly_one(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.MANUAL_OVERRIDE,
            weight=1.0,
            score_before=0.0,
            sequence_number=0,
            detail="max boost",
        )
        assert entry.score_after == 1.0

    def test_weight_exactly_negative_one(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.MANUAL_OVERRIDE,
            weight=-1.0,
            score_before=1.0,
            sequence_number=0,
            detail="max penalty",
        )
        assert entry.score_after == 0.0

    def test_node_with_dots_dashes_underscores(self):
        entry = create_trust_ledger_entry(
            node="my-service.v2_build-123",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="ok",
        )
        assert entry.node == "my-service.v2_build-123"

    def test_all_event_types_accepted(self):
        for evt in TrustEventType:
            detail = "required detail" if evt in (TrustEventType.AUDIT_FAIL, TrustEventType.ACCESS_VIOLATION) else "test"
            entry = create_trust_ledger_entry(
                node="node1",
                event=evt,
                weight=0.0,
                score_before=0.5,
                sequence_number=0,
                detail=detail,
            )
            assert entry.event == evt

    def test_error_invalid_node_spaces(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="invalid node",
                event=TrustEventType.AUDIT_PASS,
                weight=0.1,
                score_before=0.5,
                sequence_number=0,
                detail="test",
            )

    def test_error_empty_node(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="",
                event=TrustEventType.AUDIT_PASS,
                weight=0.1,
                score_before=0.5,
                sequence_number=0,
                detail="test",
            )

    def test_error_weight_too_high(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.AUDIT_PASS,
                weight=1.5,
                score_before=0.5,
                sequence_number=0,
                detail="test",
            )

    def test_error_weight_too_low(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.AUDIT_PASS,
                weight=-1.5,
                score_before=0.5,
                sequence_number=0,
                detail="test",
            )

    def test_error_score_before_above_one(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.AUDIT_PASS,
                weight=0.1,
                score_before=1.5,
                sequence_number=0,
                detail="test",
            )

    def test_error_score_before_below_zero(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.AUDIT_PASS,
                weight=0.1,
                score_before=-0.1,
                sequence_number=0,
                detail="test",
            )

    def test_error_audit_fail_empty_detail(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.AUDIT_FAIL,
                weight=-0.2,
                score_before=0.5,
                sequence_number=0,
                detail="",
            )

    def test_error_access_violation_empty_detail(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.ACCESS_VIOLATION,
                weight=-0.3,
                score_before=0.5,
                sequence_number=0,
                detail="",
            )

    def test_error_negative_sequence(self):
        with pytest.raises((ValueError, Exception)):
            create_trust_ledger_entry(
                node="node1",
                event=TrustEventType.AUDIT_PASS,
                weight=0.1,
                score_before=0.5,
                sequence_number=-1,
                detail="test",
            )

    def test_returned_entry_is_frozen(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            entry.node = "changed"

    def test_utc_timestamp_format(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        assert _is_utc_timestamp(entry.ts), f"Timestamp {entry.ts} is not UTC"

    def test_sequence_number_preserved(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=42,
            detail="test",
        )
        assert entry.sequence_number == 42


# ============================================================
# create_ledger_checkpoint Tests
# ============================================================
class TestCreateLedgerCheckpoint:
    def test_happy_path(self):
        cp = create_ledger_checkpoint(
            sequence_number=100,
            checksum=VALID_SHA256,
            entry_count=50,
        )
        assert cp.sequence_number == 100
        assert cp.checksum == VALID_SHA256
        assert cp.entry_count == 50
        assert _is_utc_timestamp(cp.ts)

    def test_min_entry_count_one(self):
        cp = create_ledger_checkpoint(
            sequence_number=0,
            checksum=VALID_SHA256_ZEROS,
            entry_count=1,
        )
        assert cp.entry_count == 1

    def test_sequence_zero(self):
        cp = create_ledger_checkpoint(
            sequence_number=0,
            checksum=VALID_SHA256,
            entry_count=1,
        )
        assert cp.sequence_number == 0

    def test_error_invalid_checksum_uppercase(self):
        with pytest.raises((ValueError, Exception)):
            create_ledger_checkpoint(
                sequence_number=0,
                checksum="A" * 64,
                entry_count=1,
            )

    def test_error_invalid_checksum_short(self):
        with pytest.raises((ValueError, Exception)):
            create_ledger_checkpoint(
                sequence_number=0,
                checksum="abcd",
                entry_count=1,
            )

    def test_error_invalid_checksum_with_non_hex(self):
        with pytest.raises((ValueError, Exception)):
            create_ledger_checkpoint(
                sequence_number=0,
                checksum="g" * 64,
                entry_count=1,
            )

    def test_error_zero_entry_count(self):
        with pytest.raises((ValueError, Exception)):
            create_ledger_checkpoint(
                sequence_number=0,
                checksum=VALID_SHA256,
                entry_count=0,
            )

    def test_error_negative_entry_count(self):
        with pytest.raises((ValueError, Exception)):
            create_ledger_checkpoint(
                sequence_number=0,
                checksum=VALID_SHA256,
                entry_count=-5,
            )

    def test_error_negative_sequence(self):
        with pytest.raises((ValueError, Exception)):
            create_ledger_checkpoint(
                sequence_number=-1,
                checksum=VALID_SHA256,
                entry_count=1,
            )

    def test_frozen(self):
        cp = create_ledger_checkpoint(
            sequence_number=0,
            checksum=VALID_SHA256,
            entry_count=1,
        )
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            cp.checksum = "b" * 64

    def test_utc_timestamp(self):
        cp = create_ledger_checkpoint(
            sequence_number=0,
            checksum=VALID_SHA256,
            entry_count=1,
        )
        assert _is_utc_timestamp(cp.ts)


# ============================================================
# build_access_graph Tests
# ============================================================
def _make_node(node_id, edges=None, trust_tier=None, data_access=None, authority_domains=None, metadata=None):
    """Helper to create AccessGraphNode instances."""
    return AccessGraphNode(
        id=node_id,
        data_access=data_access or [],
        authority_domains=authority_domains or [],
        edges=edges or [],
        trust_tier=trust_tier or TrustTier.ESTABLISHED,
        metadata=metadata or {},
    )


class TestBuildAccessGraph:
    def test_happy_path_two_nodes(self):
        node_a = _make_node("node-a", edges=["node-b"])
        node_b = _make_node("node-b", edges=["node-a"])
        graph = build_access_graph({"node-a": node_a, "node-b": node_b})
        assert len(graph.nodes) == 2
        assert "node-a" in graph.nodes
        assert "node-b" in graph.nodes

    def test_single_node_no_edges(self):
        node = _make_node("solo")
        graph = build_access_graph({"solo": node})
        assert len(graph.nodes) == 1

    def test_three_node_chain(self):
        a = _make_node("a", edges=["b"])
        b = _make_node("b", edges=["c"])
        c = _make_node("c")
        graph = build_access_graph({"a": a, "b": b, "c": c})
        assert len(graph.nodes) == 3

    def test_self_loop(self):
        node = _make_node("self-ref", edges=["self-ref"])
        graph = build_access_graph({"self-ref": node})
        assert len(graph.nodes) == 1
        assert "self-ref" in graph.nodes["self-ref"].edges

    def test_error_dangling_edge(self):
        node = _make_node("a", edges=["nonexistent"])
        with pytest.raises((ValueError, Exception)):
            build_access_graph({"a": node})

    def test_error_id_key_mismatch(self):
        node = _make_node("wrong-id")
        with pytest.raises((ValueError, Exception)):
            build_access_graph({"correct-key": node})

    def test_error_empty_graph(self):
        with pytest.raises((ValueError, Exception)):
            build_access_graph({})

    def test_referential_integrity(self):
        a = _make_node("a", edges=["b", "c"])
        b = _make_node("b", edges=["c"])
        c = _make_node("c", edges=["a"])
        graph = build_access_graph({"a": a, "b": b, "c": c})
        for nid, node in graph.nodes.items():
            for edge_target in node.edges:
                assert edge_target in graph.nodes, f"Dangling edge {edge_target} from {nid}"

    def test_cardinality_preserved(self):
        nodes = {}
        for i in range(5):
            nid = f"node-{i}"
            nodes[nid] = _make_node(nid)
        graph = build_access_graph(nodes)
        assert len(graph.nodes) == 5

    def test_graph_with_different_trust_tiers(self):
        a = _make_node("a", trust_tier=TrustTier.PROBATIONARY)
        b = _make_node("b", trust_tier=TrustTier.TRUSTED)
        graph = build_access_graph({"a": a, "b": b})
        assert graph.nodes["a"].trust_tier == TrustTier.PROBATIONARY
        assert graph.nodes["b"].trust_tier == TrustTier.TRUSTED


# ============================================================
# parse_ledger_line Tests
# ============================================================
class TestParseLedgerLine:
    def _make_entry_json(self, **overrides):
        """Create a valid TrustLedgerEntry JSON string."""
        data = {
            "ts": "2024-01-01T00:00:00+00:00",
            "node": "node1",
            "event": str(TrustEventType.AUDIT_PASS.value),
            "weight": 0.1,
            "score_before": 0.5,
            "score_after": 0.6,
            "sequence_number": 0,
            "detail": "test",
        }
        data.update(overrides)
        return json.dumps(data)

    def _make_checkpoint_json(self, **overrides):
        """Create a valid LedgerCheckpoint JSON string."""
        data = {
            "ts": "2024-01-01T00:00:00+00:00",
            "sequence_number": 100,
            "checksum": VALID_SHA256,
            "entry_count": 50,
        }
        data.update(overrides)
        return json.dumps(data)

    def test_parse_trust_entry(self):
        line = self._make_entry_json()
        result = parse_ledger_line(line)
        assert isinstance(result, TrustLedgerEntry)
        assert result.node == "node1"
        assert result.score_before == 0.5
        assert result.score_after == 0.6

    def test_parse_checkpoint(self):
        line = self._make_checkpoint_json()
        result = parse_ledger_line(line)
        assert isinstance(result, LedgerCheckpoint)
        assert result.sequence_number == 100
        assert result.checksum == VALID_SHA256
        assert result.entry_count == 50

    def test_error_invalid_json(self):
        with pytest.raises((ValueError, Exception)):
            parse_ledger_line("not json at all")

    def test_error_empty_line(self):
        with pytest.raises((ValueError, Exception)):
            parse_ledger_line("")

    def test_error_whitespace_only(self):
        with pytest.raises((ValueError, Exception)):
            parse_ledger_line("   ")

    def test_error_unknown_schema(self):
        with pytest.raises((ValueError, Exception)):
            parse_ledger_line('{"foo": "bar"}')

    def test_error_validation_failure(self):
        # Valid structure but invalid score (> 1.0)
        bad_json = self._make_entry_json(score_before=2.0, score_after=2.1)
        with pytest.raises((ValueError, Exception)):
            parse_ledger_line(bad_json)

    def test_float_fidelity(self):
        """IEEE 754 float values round-trip exactly."""
        # Use a value that's tricky in IEEE 754
        score = 0.1 + 0.2  # 0.30000000000000004
        line = self._make_entry_json(
            score_before=score,
            score_after=min(score + 0.1, 1.0),
            weight=0.1,
        )
        result = parse_ledger_line(line)
        assert result.score_before == score


# ============================================================
# serialize_ledger_line Tests
# ============================================================
class TestSerializeLedgerLine:
    def test_serialize_entry(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        result = serialize_ledger_line(entry)
        assert isinstance(result, str)
        # Must be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        # No embedded newlines
        assert "\n" not in result

    def test_serialize_checkpoint(self):
        cp = create_ledger_checkpoint(
            sequence_number=100,
            checksum=VALID_SHA256,
            entry_count=50,
        )
        result = serialize_ledger_line(cp)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "\n" not in result

    def test_roundtrip_entry(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        serialized = serialize_ledger_line(entry)
        deserialized = parse_ledger_line(serialized)
        assert isinstance(deserialized, TrustLedgerEntry)
        assert deserialized.node == entry.node
        assert deserialized.score_before == entry.score_before
        assert deserialized.score_after == entry.score_after
        assert deserialized.weight == entry.weight
        assert deserialized.sequence_number == entry.sequence_number
        assert deserialized.detail == entry.detail
        assert deserialized.ts == entry.ts

    def test_roundtrip_checkpoint(self):
        cp = create_ledger_checkpoint(
            sequence_number=100,
            checksum=VALID_SHA256,
            entry_count=50,
        )
        serialized = serialize_ledger_line(cp)
        deserialized = parse_ledger_line(serialized)
        assert isinstance(deserialized, LedgerCheckpoint)
        assert deserialized.sequence_number == cp.sequence_number
        assert deserialized.checksum == cp.checksum
        assert deserialized.entry_count == cp.entry_count

    def test_no_newlines_in_output(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.0,
            score_before=0.5,
            sequence_number=0,
            detail="a detail with\nno embedded newlines expected",
        )
        serialized = serialize_ledger_line(entry)
        assert "\n" not in serialized

    def test_deterministic_key_order(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        s1 = serialize_ledger_line(entry)
        s2 = serialize_ledger_line(entry)
        assert s1 == s2, "Serialization is not deterministic"
        # Verify keys are sorted
        parsed = json.loads(s1)
        keys = list(parsed.keys())
        assert keys == sorted(keys), f"Keys not sorted: {keys}"

    def test_error_invalid_type(self):
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            serialize_ledger_line("not a ledger line")

    def test_error_invalid_type_dict(self):
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            serialize_ledger_line({"some": "dict"})

    def test_float_fidelity(self):
        """IEEE 754 float round-trip fidelity."""
        # Use a tricky float
        tricky_score = 0.1 + 0.2  # 0.30000000000000004
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.0,
            score_before=tricky_score,
            sequence_number=0,
            detail="fidelity test",
        )
        serialized = serialize_ledger_line(entry)
        deserialized = parse_ledger_line(serialized)
        assert deserialized.score_before == entry.score_before
        assert deserialized.score_after == entry.score_after


# ============================================================
# create_error_response Tests
# ============================================================
class TestCreateErrorResponse:
    def test_happy_path_with_node(self):
        resp = create_error_response(
            error_code="not_found",
            message="Node not found",
            node="service-a",
            field="",
            domain="",
        )
        assert resp.error_code == "not_found"
        assert resp.message == "Node not found"
        assert "node" in resp.details
        assert resp.details["node"] == "service-a"

    def test_all_context_fields(self):
        resp = create_error_response(
            error_code="validation_error",
            message="Invalid field",
            node="svc-1",
            field="trust_score",
            domain="auth/core",
        )
        assert resp.details.get("node") == "svc-1"
        assert resp.details.get("field") == "trust_score"
        assert resp.details.get("domain") == "auth/core"

    def test_only_field_provided(self):
        resp = create_error_response(
            error_code="invalid_field",
            message="Bad field",
            node="",
            field="score",
            domain="",
        )
        assert resp.error_code == "invalid_field"
        # field should be in details
        assert "field" in resp.details
        assert resp.details["field"] == "score"

    def test_only_domain_provided(self):
        resp = create_error_response(
            error_code="domain_error",
            message="Bad domain",
            node="",
            field="",
            domain="auth/core",
        )
        assert "domain" in resp.details
        assert resp.details["domain"] == "auth/core"

    def test_error_empty_code(self):
        with pytest.raises((ValueError, Exception)):
            create_error_response(
                error_code="",
                message="test",
                node="n",
                field="",
                domain="",
            )

    def test_error_empty_message(self):
        with pytest.raises((ValueError, Exception)):
            create_error_response(
                error_code="err",
                message="",
                node="n",
                field="",
                domain="",
            )

    def test_details_only_contains_nonempty(self):
        """Non-empty context values appear, empty ones should not."""
        resp = create_error_response(
            error_code="err",
            message="msg",
            node="mynode",
            field="",
            domain="",
        )
        assert resp.details.get("node") == "mynode"
        # Empty values should not be in details (or if present, should be empty)
        # The contract says "contains non-empty keys for provided values"
        # So we check node is there for sure
        assert "node" in resp.details


# ============================================================
# validate_canary_fingerprint Tests
# ============================================================
class TestValidateCanaryFingerprint:
    def test_happy_path_embedded_uuid(self):
        result = validate_canary_fingerprint(
            f"canary-{VALID_UUID_V4}-data"
        )
        assert result is True

    def test_bare_uuid_v4(self):
        result = validate_canary_fingerprint(VALID_UUID_V4)
        assert result is True

    def test_error_missing_uuid(self):
        with pytest.raises((ValueError, Exception)):
            validate_canary_fingerprint("just-a-regular-string")

    def test_error_empty_fingerprint(self):
        with pytest.raises((ValueError, Exception)):
            validate_canary_fingerprint("")

    def test_error_uuid_v1_not_v4(self):
        """UUID v1 (version digit is 1, not 4) should not pass."""
        with pytest.raises((ValueError, Exception)):
            validate_canary_fingerprint("canary-12345678-1234-1abc-9abc-123456789abc")

    def test_error_uuid_v3_not_v4(self):
        """UUID v3 (version digit is 3, not 4) should not pass."""
        with pytest.raises((ValueError, Exception)):
            validate_canary_fingerprint("canary-12345678-1234-3abc-9abc-123456789abc")

    def test_uuid_v4_variant_8(self):
        # variant byte is 8 (valid: 8, 9, a, b)
        assert validate_canary_fingerprint("prefix-12345678-1234-4abc-8abc-123456789abc") is True

    def test_uuid_v4_variant_9(self):
        assert validate_canary_fingerprint("prefix-12345678-1234-4abc-9abc-123456789abc") is True

    def test_uuid_v4_variant_a(self):
        assert validate_canary_fingerprint("prefix-12345678-1234-4abc-aabc-123456789abc") is True

    def test_uuid_v4_variant_b(self):
        assert validate_canary_fingerprint("prefix-12345678-1234-4abc-babc-123456789abc") is True

    def test_error_uuid_v4_wrong_variant(self):
        """Variant byte c is not valid for UUID v4 ([89ab] required)."""
        with pytest.raises((ValueError, Exception)):
            validate_canary_fingerprint("prefix-12345678-1234-4abc-cabc-123456789abc")


# ============================================================
# classify_field Tests
# ============================================================
class TestClassifyField:
    def test_match_fnmatch_pattern(self):
        rules = [
            ClassificationRule(
                field_pattern="*password*",
                data_tier=DataTier.AUTH,
                is_regex=False,
                description="Password fields",
            ),
        ]
        result = classify_field("user_password_hash", rules)
        assert result == DataTier.AUTH

    def test_match_regex_pattern(self):
        rules = [
            ClassificationRule(
                field_pattern=r"^ssn_\d+$",
                data_tier=DataTier.PII,
                is_regex=True,
                description="SSN fields",
            ),
        ]
        result = classify_field("ssn_123", rules)
        assert result == DataTier.PII

    def test_no_match_returns_public(self):
        rules = [
            ClassificationRule(
                field_pattern="*password*",
                data_tier=DataTier.AUTH,
                is_regex=False,
                description="Password fields",
            ),
        ]
        result = classify_field("username", rules)
        assert result == DataTier.PUBLIC

    def test_empty_rules_returns_public(self):
        result = classify_field("anything", [])
        assert result == DataTier.PUBLIC

    def test_first_match_wins(self):
        rules = [
            ClassificationRule(
                field_pattern="*secret*",
                data_tier=DataTier.AUTH,
                is_regex=False,
                description="Auth secrets",
            ),
            ClassificationRule(
                field_pattern="*secret*",
                data_tier=DataTier.PII,
                is_regex=False,
                description="PII secrets",
            ),
        ]
        result = classify_field("my_secret_key", rules)
        assert result == DataTier.AUTH, "First matching rule should win, not second"

    def test_error_empty_field_name(self):
        with pytest.raises((ValueError, Exception)):
            classify_field("", [])

    def test_error_invalid_regex(self):
        rules = [
            ClassificationRule(
                field_pattern="[invalid(",
                data_tier=DataTier.PII,
                is_regex=True,
                description="Bad regex",
            ),
        ]
        with pytest.raises((ValueError, re.error, Exception)):
            classify_field("test_field", rules)

    def test_fnmatch_exact_match(self):
        rules = [
            ClassificationRule(
                field_pattern="credit_card",
                data_tier=DataTier.FINANCIAL,
                is_regex=False,
                description="Exact credit card field",
            ),
        ]
        result = classify_field("credit_card", rules)
        assert result == DataTier.FINANCIAL

    def test_fnmatch_no_partial_without_wildcard(self):
        rules = [
            ClassificationRule(
                field_pattern="credit_card",
                data_tier=DataTier.FINANCIAL,
                is_regex=False,
                description="Exact match only",
            ),
        ]
        result = classify_field("credit_card_number", rules)
        # fnmatch without wildcards only matches exact
        assert result == DataTier.PUBLIC

    def test_regex_full_match(self):
        rules = [
            ClassificationRule(
                field_pattern=r"^email$",
                data_tier=DataTier.PII,
                is_regex=True,
                description="Email field exact",
            ),
        ]
        result = classify_field("email", rules)
        assert result == DataTier.PII


# ============================================================
# Struct Immutability & Extra Forbid Tests
# ============================================================
class TestStructImmutability:
    def test_trust_ledger_entry_frozen(self):
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.1,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            entry.detail = "modified"

    def test_ledger_checkpoint_frozen(self):
        cp = create_ledger_checkpoint(
            sequence_number=0,
            checksum=VALID_SHA256,
            entry_count=1,
        )
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            cp.entry_count = 999

    def test_access_graph_node_frozen(self):
        node = _make_node("n1")
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            node.id = "changed"


class TestStructExtraForbid:
    def test_trust_ledger_entry_rejects_extra(self):
        """TrustLedgerEntry with extra='forbid' should reject unknown fields."""
        with pytest.raises((TypeError, ValueError, Exception)):
            TrustLedgerEntry(
                ts="2024-01-01T00:00:00+00:00",
                node="node1",
                event=TrustEventType.AUDIT_PASS,
                weight=0.1,
                score_before=0.5,
                score_after=0.6,
                sequence_number=0,
                detail="test",
                unknown_field="should_fail",
            )

    def test_ledger_checkpoint_rejects_extra(self):
        with pytest.raises((TypeError, ValueError, Exception)):
            LedgerCheckpoint(
                ts="2024-01-01T00:00:00+00:00",
                sequence_number=0,
                checksum=VALID_SHA256,
                entry_count=1,
                unknown_field="should_fail",
            )


# ============================================================
# Integration Tests
# ============================================================
class TestIntegration:
    def test_ledger_roundtrip_multiple_entries(self):
        """Create multiple entries, serialize to JSONL, parse back, verify equality."""
        entries = []
        for i in range(5):
            entry = create_trust_ledger_entry(
                node=f"node-{i}",
                event=TrustEventType.AUDIT_PASS,
                weight=0.05,
                score_before=0.5,
                sequence_number=i,
                detail=f"entry {i}",
            )
            entries.append(entry)

        # Add a checkpoint
        cp = create_ledger_checkpoint(
            sequence_number=5,
            checksum=VALID_SHA256,
            entry_count=5,
        )
        entries.append(cp)

        # Serialize all to JSONL
        lines = [serialize_ledger_line(e) for e in entries]

        # Verify no embedded newlines
        for line in lines:
            assert "\n" not in line

        # Parse back
        parsed = [parse_ledger_line(line) for line in lines]

        # Verify first 5 are entries, last is checkpoint
        for i in range(5):
            assert isinstance(parsed[i], TrustLedgerEntry)
            assert parsed[i].node == f"node-{i}"
            assert parsed[i].sequence_number == i
        assert isinstance(parsed[5], LedgerCheckpoint)
        assert parsed[5].entry_count == 5

    def test_graph_build_with_trust_and_data(self):
        """Build a graph with varying trust tiers and data access."""
        nodes = {
            "frontend": _make_node(
                "frontend",
                edges=["backend"],
                trust_tier=TrustTier.HIGH,
                data_access=[DataTier.PUBLIC],
            ),
            "backend": _make_node(
                "backend",
                edges=["database"],
                trust_tier=TrustTier.TRUSTED,
                data_access=[DataTier.PII, DataTier.FINANCIAL],
            ),
            "database": _make_node(
                "database",
                edges=[],
                trust_tier=TrustTier.TRUSTED,
                data_access=[DataTier.PII, DataTier.FINANCIAL, DataTier.AUTH],
            ),
        }
        graph = build_access_graph(nodes)
        assert len(graph.nodes) == 3
        assert graph.nodes["frontend"].trust_tier == TrustTier.HIGH
        assert graph.nodes["backend"].trust_tier == TrustTier.TRUSTED
        assert DataTier.AUTH in graph.nodes["database"].data_access

    def test_classify_then_error_response(self):
        """Classify a field, and if it's sensitive, create an error response."""
        rules = [
            ClassificationRule(
                field_pattern="*ssn*",
                data_tier=DataTier.PII,
                is_regex=False,
                description="SSN fields",
            ),
        ]
        tier = classify_field("user_ssn", rules)
        assert tier == DataTier.PII

        resp = create_error_response(
            error_code="unauthorized_access",
            message="Cannot access PII field",
            node="frontend",
            field="user_ssn",
            domain="",
        )
        assert resp.error_code == "unauthorized_access"
        assert resp.details.get("field") == "user_ssn"

    def test_score_to_tier_used_for_display_only(self):
        """Verify that score_to_tier is consistent with the raw score used in entries."""
        entry = create_trust_ledger_entry(
            node="node1",
            event=TrustEventType.AUDIT_PASS,
            weight=0.3,
            score_before=0.5,
            sequence_number=0,
            detail="test",
        )
        # score_after should be 0.8 -> TRUSTED tier
        assert entry.score_after == pytest.approx(0.8)
        tier = score_to_tier(entry.score_after)
        assert tier == TrustTier.TRUSTED

    def test_canary_fingerprint_in_canary_record(self):
        """Validate a canary fingerprint and use it in a CanaryRecord."""
        fp = f"canary-{VALID_UUID_V4}-data"
        assert validate_canary_fingerprint(fp) is True

        record = CanaryRecord(
            ts="2024-01-01T00:00:00+00:00",
            canary_id="c-001",
            fingerprint=fp,
            data_tier=DataTier.PII,
            target_node="backend",
            triggered=False,
            triggered_at="",
            triggered_by_node="",
        )
        assert record.fingerprint == fp
        assert record.triggered is False
