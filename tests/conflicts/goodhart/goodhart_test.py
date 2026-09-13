"""
Adversarial hidden acceptance tests for the Conflict Resolver component.

These tests catch implementations that 'teach to the test' by hardcoding
returns or using shortcuts that pass visible tests without truly satisfying
the contract.
"""

import pytest
import json
import uuid
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone
from dateutil.parser import isoparse

from arbiter.conflicts import *


# ─── Helpers ───────────────────────────────────────────────────────────────────

def make_config(**overrides):
    """Create a valid ConflictResolverConfig with sensible defaults."""
    defaults = {
        "window_timeout_seconds": 1.0,
        "authority_override_floor": 0.5,
        "trust_delta_threshold": 0.1,
        "checkpoint_interval": 10,
        "conflict_log_path": "/tmp/test_conflict_log.jsonl",
        "protected_tiers": ["critical", "pii", "pii_critical", "financial"],
    }
    defaults.update(overrides)
    return ConflictResolverConfig(**defaults)


def make_node_value(node_id, value, trust, authoritative=False, span_id=None):
    """Create a NodeValue for testing."""
    return NodeValue(
        node_id=node_id,
        value_serialized=value,
        trust_score_snapshot=trust,
        is_authoritative=authoritative,
        span_id=span_id or f"span-{node_id}",
    )


def make_conflict_record(
    competing_values,
    status=ResolutionStatus.DETECTED,
    data_tier="standard",
    domain="test-domain",
    field="test-field",
    execution_id=None,
    conflict_id=None,
    resolution=None,
    blocks_deploy=False,
):
    """Create a ConflictRecord for testing."""
    return ConflictRecord(
        conflict_id=conflict_id or str(uuid.uuid4()),
        execution_id=execution_id or str(uuid.uuid4()),
        domain=domain,
        field=field,
        data_tier=data_tier,
        competing_values=competing_values,
        detected_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        resolution=resolution,
        blocks_deploy=blocks_deploy,
    )


def make_span_report(node_id, value, domain="test-domain", field="test-field",
                     execution_id="exec-001", span_id=None):
    """Create a SpanFieldReport for testing."""
    return SpanFieldReport(
        span_id=span_id or f"span-{node_id}-{uuid.uuid4().hex[:8]}",
        execution_id=execution_id,
        node_id=node_id,
        domain=domain,
        field=field,
        value_serialized=value,
        reported_at=datetime.now(timezone.utc).isoformat(),
    )


def make_mock_store(records=None):
    """Create a mock ConflictStore."""
    store = MagicMock(spec=ConflictStore)
    stored = list(records or [])
    store.append = MagicMock(side_effect=lambda r: stored.append(r))
    store.load_all = MagicMock(return_value=stored)
    store.verify_checksums = MagicMock(return_value=True)
    return store, stored


def make_mock_trust_lookup(scores=None):
    """Create a mock TrustLookup. scores is a dict of node_id -> trust_score."""
    lookup = MagicMock(spec=TrustLookup)
    scores = scores or {}
    lookup.lookup_trust_score = MagicMock(
        side_effect=lambda node_id: scores.get(node_id, 0.5)
    )
    return lookup


def make_mock_authority_lookup(authorities=None):
    """Create a mock AuthorityLookup. authorities is a set of authoritative node_ids."""
    lookup = MagicMock(spec=AuthorityLookup)
    authorities = authorities or set()
    lookup.is_authoritative = MagicMock(
        side_effect=lambda node_id: node_id in authorities
    )
    return lookup


def make_mock_signal_emitter():
    """Create a mock SignalEmitter."""
    emitter = MagicMock(spec=SignalEmitter)
    emitter.emit_signal = MagicMock()
    return emitter


# ─── resolve() tests ──────────────────────────────────────────────────────────

class TestGoodhartResolve:

    def test_goodhart_resolve_authority_trust_just_above_floor(self):
        """Authority resolution should succeed when the authoritative node's trust is strictly above floor by a tiny epsilon."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        auth_node = make_node_value("auth-node", '"val_a"', 0.501, authoritative=True)
        other_node = make_node_value("other-node", '"val_b"', 0.8, authoritative=False)
        conflict = make_conflict_record(
            competing_values=[auth_node, other_node],
            data_tier="standard",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        # Construct resolver with dependencies
        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        assert result.resolution.strategy == ResolutionStrategy.AUTHORITY
        assert result.resolution.winner_node_id == "auth-node"

    def test_goodhart_resolve_authority_exactly_at_floor_rejects(self):
        """Authority step must be skipped when authoritative node trust equals exactly the floor (not strictly greater)."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.8)
        auth_node = make_node_value("auth-node", '"val_a"', 0.5, authoritative=True)
        other_node = make_node_value("other-node", '"val_b"', 0.49, authoritative=False)
        conflict = make_conflict_record(
            competing_values=[auth_node, other_node],
            data_tier="standard",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Authority step should be skipped, so NOT AUTHORITY_RESOLVED
        assert result.status != ResolutionStatus.AUTHORITY_RESOLVED
        assert result.status in (ResolutionStatus.TRUST_RESOLVED, ResolutionStatus.UNRESOLVABLE)

    def test_goodhart_resolve_trust_delta_just_above_threshold(self):
        """Trust arbitration succeeds when delta is just above the threshold."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        node_a = make_node_value("node-a", '"val_a"', 0.8, authoritative=False)
        node_b = make_node_value("node-b", '"val_b"', 0.699, authoritative=False)
        conflict = make_conflict_record(
            competing_values=[node_a, node_b],
            data_tier="standard",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Delta = 0.101, threshold = 0.1, so should be TRUST_RESOLVED
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.strategy == ResolutionStrategy.TRUST_ARBITRATION
        assert result.resolution.winner_node_id == "node-a"

    def test_goodhart_resolve_trust_delta_just_below_threshold(self):
        """Trust arbitration fails when delta is just below the threshold, resulting in UNRESOLVABLE."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        node_a = make_node_value("node-a", '"val_a"', 0.8, authoritative=False)
        node_b = make_node_value("node-b", '"val_b"', 0.701, authoritative=False)
        conflict = make_conflict_record(
            competing_values=[node_a, node_b],
            data_tier="standard",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Delta = 0.099, threshold = 0.1, so should be UNRESOLVABLE
        assert result.status == ResolutionStatus.UNRESOLVABLE
        assert result.resolution.strategy == ResolutionStrategy.HUMAN

    def test_goodhart_resolve_nway_5nodes_authority_one(self):
        """In a 5-way conflict with exactly one authoritative node above floor, authority resolution should apply."""
        config = make_config(authority_override_floor=0.3, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("node-1", '"v1"', 0.4, authoritative=True),
            make_node_value("node-2", '"v2"', 0.9, authoritative=False),
            make_node_value("node-3", '"v3"', 0.85, authoritative=False),
            make_node_value("node-4", '"v4"', 0.7, authoritative=False),
            make_node_value("node-5", '"v5"', 0.6, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        assert result.resolution.winner_node_id == "node-1"
        assert len(conflict.competing_values) == 5

    def test_goodhart_resolve_nway_trust_ignores_3rd_and_beyond(self):
        """Trust arbitration delta is computed only from top-2 trust scores; 3rd+ scores are irrelevant."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.2)
        nodes = [
            make_node_value("node-top", '"v1"', 0.9, authoritative=False),
            make_node_value("node-2nd", '"v2"', 0.5, authoritative=False),
            make_node_value("node-3rd", '"v3"', 0.5, authoritative=False),
            make_node_value("node-4th", '"v4"', 0.3, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Delta between top-2 = 0.9 - 0.5 = 0.4 > 0.2 threshold
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.winner_node_id == "node-top"

    def test_goodhart_resolve_two_authoritative_nodes_skips_authority(self):
        """When two nodes are authoritative, authority step is skipped (requires exactly one)."""
        config = make_config(authority_override_floor=0.3, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("auth-1", '"v1"', 0.9, authoritative=True),
            make_node_value("auth-2", '"v2"', 0.5, authoritative=True),
            make_node_value("regular", '"v3"', 0.3, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Authority skipped, trust delta between 0.9 and 0.5 = 0.4 > 0.1
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.strategy == ResolutionStrategy.TRUST_ARBITRATION

    def test_goodhart_resolve_unresolvable_protected_tier_blocks(self):
        """UNRESOLVABLE conflict with data_tier in protected_tiers sets blocks_deploy=True."""
        config = make_config(
            authority_override_floor=0.5,
            trust_delta_threshold=0.5,
            protected_tiers=["pii_critical", "financial"],
        )
        nodes = [
            make_node_value("n1", '"v1"', 0.6, authoritative=False),
            make_node_value("n2", '"v2"', 0.55, authoritative=False),
        ]
        conflict = make_conflict_record(
            competing_values=nodes,
            data_tier="pii_critical",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE
        assert result.blocks_deploy is True

    def test_goodhart_resolve_unresolvable_signal_fields_complete(self):
        """ConflictSignal emitted for unresolvable conflict has all required fields populated correctly."""
        config = make_config(
            authority_override_floor=0.5,
            trust_delta_threshold=0.5,
            protected_tiers=["critical"],
        )
        nodes = [
            make_node_value("n-alpha", '"val_alpha"', 0.82, authoritative=False),
            make_node_value("n-beta", '"val_beta"', 0.79, authoritative=False),
            make_node_value("n-gamma", '"val_gamma"', 0.50, authoritative=False),
        ]
        conflict = make_conflict_record(
            competing_values=nodes,
            data_tier="critical",
            domain="signals-domain",
            field="signals-field",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE
        emitter.emit_signal.assert_called_once()
        signal = emitter.emit_signal.call_args[0][0] if emitter.emit_signal.call_args[0] else emitter.emit_signal.call_args[1].get("signal")

        assert signal.conflict_id == conflict.conflict_id
        assert signal.execution_id == conflict.execution_id
        assert set(signal.competing_node_ids) == {"n-alpha", "n-beta", "n-gamma"}
        assert abs(signal.max_trust_score - 0.82) < 0.001
        assert signal.blocks_deploy is True

    def test_goodhart_resolve_rationale_contains_all_threshold_values(self):
        """Resolution rationale must include threshold values from config, not just node info."""
        config = make_config(authority_override_floor=0.35, trust_delta_threshold=0.1)
        auth_node = make_node_value("auth-node-xyz", '"val_a"', 0.7, authoritative=True)
        other_node = make_node_value("other-node-abc", '"val_b"', 0.4, authoritative=False)
        conflict = make_conflict_record(competing_values=[auth_node, other_node])

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        rationale = result.resolution.rationale
        # Rationale must include the authoritative node ID
        assert "auth-node-xyz" in rationale
        # Rationale must include relevant threshold or trust values
        assert "0.35" in rationale or "authority_override_floor" in rationale
        assert "0.7" in rationale

    def test_goodhart_resolve_all_zero_trust_scores(self):
        """All-zero trust scores produce delta=0 which is below any positive threshold: UNRESOLVABLE."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("n1", '"v1"', 0.0, authoritative=False),
            make_node_value("n2", '"v2"', 0.0, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE

    def test_goodhart_resolve_equal_trust_scores_unresolvable(self):
        """Identical trust scores produce delta=0 which is below any positive threshold: UNRESOLVABLE."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.05)
        nodes = [
            make_node_value("n1", '"v1"', 0.75, authoritative=False),
            make_node_value("n2", '"v2"', 0.75, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE

    def test_goodhart_resolve_authority_below_floor_and_trust_delta_sufficient(self):
        """Auth node below floor should skip to Step 2, and if trust delta is sufficient, TRUST_RESOLVED (not directly UNRESOLVABLE)."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("auth-low", '"v1"', 0.3, authoritative=True),
            make_node_value("high-trust", '"v2"', 0.9, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Delta = 0.6 > 0.1, should be trust resolved
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.strategy == ResolutionStrategy.TRUST_ARBITRATION
        assert result.resolution.winner_node_id == "high-trust"

    def test_goodhart_resolve_no_signal_emitted_for_authority_resolved(self):
        """No ConflictSignal should be emitted for authority-resolved conflicts."""
        config = make_config(authority_override_floor=0.3, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("auth-node", '"v1"', 0.8, authoritative=True),
            make_node_value("other", '"v2"', 0.4, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        emitter.emit_signal.assert_not_called()

    def test_goodhart_resolve_no_signal_emitted_for_trust_resolved(self):
        """No ConflictSignal should be emitted for trust-resolved conflicts."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("n1", '"v1"', 0.9, authoritative=False),
            make_node_value("n2", '"v2"', 0.3, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.TRUST_RESOLVED
        emitter.emit_signal.assert_not_called()

    def test_goodhart_resolve_resolved_at_is_utc_iso8601(self):
        """The resolved_at timestamp must be a valid UTC ISO-8601 string."""
        config = make_config(authority_override_floor=0.3, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("auth", '"v1"', 0.8, authoritative=True),
            make_node_value("other", '"v2"', 0.4, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        ts = result.resolution.resolved_at
        parsed = isoparse(ts)
        # Must be UTC
        assert parsed.tzinfo is not None or ts.endswith("Z")
        if parsed.tzinfo is not None:
            assert parsed.utcoffset().total_seconds() == 0

    def test_goodhart_resolve_blocks_deploy_false_for_non_unresolvable(self):
        """Authority/trust resolved conflicts should NOT have blocks_deploy=True even if data_tier is protected."""
        config = make_config(
            authority_override_floor=0.3,
            trust_delta_threshold=0.1,
            protected_tiers=["critical"],
        )
        nodes = [
            make_node_value("auth", '"v1"', 0.8, authoritative=True),
            make_node_value("other", '"v2"', 0.4, authoritative=False),
        ]
        conflict = make_conflict_record(
            competing_values=nodes,
            data_tier="critical",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        assert result.blocks_deploy is False

    def test_goodhart_resolve_winner_matches_highest_trust_novel_scores(self):
        """In trust arbitration with novel trust scores, the winner must be the highest-trust node."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        nodes = [
            make_node_value("node-low", '"v1"', 0.33, authoritative=False),
            make_node_value("node-mid", '"v2"', 0.67, authoritative=False),
            make_node_value("node-high", '"v3"', 0.99, authoritative=False),
        ]
        conflict = make_conflict_record(competing_values=nodes)

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        # Delta between top-2 = 0.99 - 0.67 = 0.32 > 0.1
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.winner_node_id == "node-high"

    def test_goodhart_resolve_error_already_resolved_human_reviewed(self):
        """A HUMAN_REVIEWED conflict must raise already_resolved if resolve() is called on it."""
        config = make_config()
        nodes = [
            make_node_value("n1", '"v1"', 0.8),
            make_node_value("n2", '"v2"', 0.4),
        ]
        resolution = Resolution(
            strategy=ResolutionStrategy.HUMAN,
            winner_node_id="n1",
            resolved_at=datetime.now(timezone.utc).isoformat(),
            rationale="Human decided",
            reviewed_by="admin",
        )
        conflict = make_conflict_record(
            competing_values=nodes,
            status=ResolutionStatus.HUMAN_REVIEWED,
            resolution=resolution,
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        with pytest.raises(Exception) as exc_info:
            resolver.resolve(conflict)
        # Should be an already_resolved error
        error_str = str(exc_info.value).lower()
        assert "already" in error_str or "resolved" in error_str or "CONFLICT_ALREADY_RESOLVED" in str(exc_info.value)

    def test_goodhart_resolve_signal_trust_delta_accuracy(self):
        """ConflictSignal trust_delta must accurately reflect the difference between top-2 trust scores."""
        config = make_config(
            authority_override_floor=0.5,
            trust_delta_threshold=0.5,
            protected_tiers=["critical"],
        )
        nodes = [
            make_node_value("n1", '"v1"', 0.82, authoritative=False),
            make_node_value("n2", '"v2"', 0.79, authoritative=False),
            make_node_value("n3", '"v3"', 0.50, authoritative=False),
        ]
        conflict = make_conflict_record(
            competing_values=nodes,
            data_tier="critical",
        )

        store, stored = make_mock_store()
        emitter = make_mock_signal_emitter()

        try:
            resolver = ConflictResolver(config=config, store=store, signal_emitter=emitter)
        except TypeError:
            resolver = ConflictResolver(config, store, emitter)

        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE
        emitter.emit_signal.assert_called_once()
        signal = emitter.emit_signal.call_args[0][0] if emitter.emit_signal.call_args[0] else emitter.emit_signal.call_args[1].get("signal")

        assert abs(signal.trust_delta - 0.03) < 0.001
        assert abs(signal.max_trust_score - 0.82) < 0.001


# ─── ingest() tests ───────────────────────────────────────────────────────────

class TestGoodhartIngest:

    def test_goodhart_ingest_three_distinct_values_conflict(self):
        """Three different nodes with three distinct values should produce a conflict with 3 competing_values."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"n1": 0.8, "n2": 0.6, "n3": 0.4})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        import time

        span1 = make_span_report("n1", '"val_1"')
        span2 = make_span_report("n2", '"val_2"')
        span3 = make_span_report("n3", '"val_3"')

        detector.ingest(span1)
        detector.ingest(span2)
        detector.ingest(span3)

        time.sleep(0.15)

        # Trigger window close with a new span in a different key
        trigger = make_span_report("n4", '"trigger"', domain="other-domain")
        conflicts = detector.ingest(trigger)

        # If lazy timeout didn't trigger, try flush
        if not conflicts:
            conflicts = detector.flush()

        assert len(conflicts) >= 1
        # Find the conflict for our test domain
        target = [c for c in conflicts if c.domain == "test-domain"]
        assert len(target) == 1
        assert len(target[0].competing_values) == 3
        assert target[0].status == ResolutionStatus.DETECTED

    def test_goodhart_ingest_same_node_different_spans_both_kept(self):
        """Same node_id but different span_ids should both be kept (dedup is by node_id+span_id pair)."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"node-x": 0.7})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        span1 = make_span_report("node-x", '"val_1"', span_id="span-aaa")
        span2 = make_span_report("node-x", '"val_2"', span_id="span-bbb")

        detector.ingest(span1)
        detector.ingest(span2)

        import time
        time.sleep(0.15)

        conflicts = detector.flush()

        target = [c for c in conflicts if c.domain == "test-domain"]
        assert len(target) == 1
        assert len(target[0].competing_values) == 2

    def test_goodhart_ingest_different_nodes_same_span_id_both_kept(self):
        """Different node_ids with the same span_id should both be kept since dedup is by (node_id, span_id) pair."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"node-a": 0.7, "node-b": 0.6})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        span1 = make_span_report("node-a", '"val_1"', span_id="shared-span-id")
        span2 = make_span_report("node-b", '"val_2"', span_id="shared-span-id")

        detector.ingest(span1)
        detector.ingest(span2)

        conflicts = detector.flush()

        target = [c for c in conflicts if c.domain == "test-domain"]
        assert len(target) == 1
        assert len(target[0].competing_values) == 2

    def test_goodhart_ingest_distinct_values_required_not_just_distinct_nodes(self):
        """Multiple nodes with identical values should not produce a conflict."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"n1": 0.8, "n2": 0.6, "n3": 0.4})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        # Three nodes, all same value
        span1 = make_span_report("n1", '"same_value"')
        span2 = make_span_report("n2", '"same_value"')
        span3 = make_span_report("n3", '"same_value"')

        detector.ingest(span1)
        detector.ingest(span2)
        detector.ingest(span3)

        conflicts = detector.flush()

        target = [c for c in conflicts if c.domain == "test-domain"]
        assert len(target) == 0

    def test_goodhart_ingest_window_key_includes_execution_id(self):
        """Spans with different execution_ids should be in separate windows."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"n1": 0.8, "n2": 0.6})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        span1 = make_span_report("n1", '"val_1"', execution_id="exec-A")
        span2 = make_span_report("n2", '"val_2"', execution_id="exec-B")

        detector.ingest(span1)
        detector.ingest(span2)

        conflicts = detector.flush()

        # Each window has only 1 distinct span, so no conflicts
        assert len(conflicts) == 0

    def test_goodhart_ingest_conflict_detected_at_is_utc_iso8601(self):
        """detected_at on newly detected conflicts must be valid UTC ISO-8601."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"n1": 0.8, "n2": 0.6})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        span1 = make_span_report("n1", '"val_1"')
        span2 = make_span_report("n2", '"val_2"')

        detector.ingest(span1)
        detector.ingest(span2)

        conflicts = detector.flush()
        assert len(conflicts) >= 1

        ts = conflicts[0].detected_at
        parsed = isoparse(ts)
        assert parsed.tzinfo is not None or ts.endswith("Z")
        if parsed.tzinfo is not None:
            assert parsed.utcoffset().total_seconds() == 0

    def test_goodhart_ingest_execution_id_field_preserved(self):
        """Conflict record must preserve the exact execution_id, domain, field from the span reports."""
        config = make_config(window_timeout_seconds=0.1)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"n1": 0.8, "n2": 0.6})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        span1 = make_span_report(
            "n1", '"val_1"',
            domain="exotic.domain.v2",
            field="nested.field.path",
            execution_id="exec-zzzz-9999",
        )
        span2 = make_span_report(
            "n2", '"val_2"',
            domain="exotic.domain.v2",
            field="nested.field.path",
            execution_id="exec-zzzz-9999",
        )

        detector.ingest(span1)
        detector.ingest(span2)

        conflicts = detector.flush()
        assert len(conflicts) >= 1

        c = conflicts[0]
        assert c.execution_id == "exec-zzzz-9999"
        assert c.domain == "exotic.domain.v2"
        assert c.field == "nested.field.path"


# ─── flush() tests ────────────────────────────────────────────────────────────

class TestGoodhartFlush:

    def test_goodhart_flush_multiple_windows_mixed(self):
        """Flush with multiple open windows: only windows with 2+ distinct values produce conflicts."""
        config = make_config(window_timeout_seconds=10.0)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({
            "n1": 0.8, "n2": 0.6, "n3": 0.4, "n4": 0.5, "n5": 0.7
        })
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        # Window 1: 2 distinct values (conflict)
        detector.ingest(make_span_report("n1", '"v1"', domain="dom-a", field="f1", execution_id="e1"))
        detector.ingest(make_span_report("n2", '"v2"', domain="dom-a", field="f1", execution_id="e1"))

        # Window 2: 3 distinct values (conflict)
        detector.ingest(make_span_report("n3", '"v3"', domain="dom-b", field="f2", execution_id="e2"))
        detector.ingest(make_span_report("n4", '"v4"', domain="dom-b", field="f2", execution_id="e2"))
        detector.ingest(make_span_report("n5", '"v5"', domain="dom-b", field="f2", execution_id="e2"))

        # Window 3: 1 value only (no conflict)
        detector.ingest(make_span_report("n1", '"v1"', domain="dom-c", field="f3", execution_id="e3"))

        conflicts = detector.flush()

        assert len(conflicts) == 2
        assert all(c.status == ResolutionStatus.DETECTED for c in conflicts)
        assert all(len(c.competing_values) >= 2 for c in conflicts)

    def test_goodhart_flush_second_flush_empty(self):
        """Second flush after first flush should return empty list since buffer was emptied."""
        config = make_config(window_timeout_seconds=10.0)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({"n1": 0.8, "n2": 0.6})
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        detector.ingest(make_span_report("n1", '"v1"'))
        detector.ingest(make_span_report("n2", '"v2"'))

        first_flush = detector.flush()
        assert len(first_flush) >= 1

        second_flush = detector.flush()
        assert len(second_flush) == 0

    def test_goodhart_ingest_conflict_id_unique_per_conflict(self):
        """Each detected conflict should have a unique conflict_id, even from the same flush."""
        config = make_config(window_timeout_seconds=10.0)
        store, stored = make_mock_store()
        trust_lookup = make_mock_trust_lookup({
            "n1": 0.8, "n2": 0.6, "n3": 0.7, "n4": 0.5
        })
        authority_lookup = make_mock_authority_lookup(set())

        try:
            detector = ConflictDetector(
                config=config,
                store=store,
                trust_lookup=trust_lookup,
                authority_lookup=authority_lookup,
            )
        except (TypeError, NameError):
            pytest.skip("ConflictDetector constructor signature differs from expected")

        detector.ingest(make_span_report("n1", '"v1"', domain="dom-a", execution_id="e1"))
        detector.ingest(make_span_report("n2", '"v2"', domain="dom-a", execution_id="e1"))
        detector.ingest(make_span_report("n3", '"v3"', domain="dom-b", execution_id="e2"))
        detector.ingest(make_span_report("n4", '"v4"', domain="dom-b", execution_id="e2"))

        conflicts = detector.flush()
        assert len(conflicts) == 2
        assert conflicts[0].conflict_id != conflicts[1].conflict_id


# ─── submit_human_review() tests ──────────────────────────────────────────────

class TestGoodhartSubmitHumanReview:

    def test_goodhart_submit_human_review_rationale_preserved(self):
        """The rationale provided to submit_human_review should be preserved in the returned record."""
        unresolvable = make_conflict_record(
            competing_values=[
                make_node_value("n1", '"v1"', 0.5),
                make_node_value("n2", '"v2"', 0.5),
            ],
            status=ResolutionStatus.UNRESOLVABLE,
            conflict_id="conflict-review-test",
            resolution=Resolution(
                strategy=ResolutionStrategy.HUMAN,
                winner_node_id="",
                resolved_at=datetime.now(timezone.utc).isoformat(),
                rationale="Unresolvable",
                reviewed_by="",
            ),
            blocks_deploy=True,
        )

        store, stored = make_mock_store([unresolvable])

        try:
            reviewer = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            reviewer = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        custom_rationale = "After manual inspection, node n1 has the correct value based on source data."
        result = reviewer.submit_human_review(
            conflict_id="conflict-review-test",
            winner_node_id="n1",
            reviewed_by="admin@example.com",
            rationale=custom_rationale,
        )

        assert result.status == ResolutionStatus.HUMAN_REVIEWED
        assert result.resolution.rationale == custom_rationale
        assert result.resolution.reviewed_by == "admin@example.com"
        assert result.resolution.winner_node_id == "n1"
        assert result.resolution.strategy == ResolutionStrategy.HUMAN

    def test_goodhart_submit_human_review_authority_resolved_rejects(self):
        """Attempting human review on an AUTHORITY_RESOLVED conflict must fail (terminal state)."""
        resolved = make_conflict_record(
            competing_values=[
                make_node_value("n1", '"v1"', 0.8, authoritative=True),
                make_node_value("n2", '"v2"', 0.4),
            ],
            status=ResolutionStatus.AUTHORITY_RESOLVED,
            conflict_id="conflict-auth-resolved",
            resolution=Resolution(
                strategy=ResolutionStrategy.AUTHORITY,
                winner_node_id="n1",
                resolved_at=datetime.now(timezone.utc).isoformat(),
                rationale="Authority resolved",
                reviewed_by="",
            ),
        )

        store, stored = make_mock_store([resolved])

        try:
            reviewer = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            reviewer = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        with pytest.raises(Exception) as exc_info:
            reviewer.submit_human_review(
                conflict_id="conflict-auth-resolved",
                winner_node_id="n1",
                reviewed_by="admin",
                rationale="Override",
            )
        error_str = str(exc_info.value).lower()
        assert "status" in error_str or "invalid" in error_str or "already" in error_str


# ─── get_unresolved() tests ───────────────────────────────────────────────────

class TestGoodhartGetUnresolved:

    def _make_records_for_filtering(self):
        """Create a set of records with various statuses for filtering tests."""
        records = []
        for status, domain in [
            (ResolutionStatus.DETECTED, "dom-a"),
            (ResolutionStatus.UNRESOLVABLE, "dom-a"),
            (ResolutionStatus.AUTHORITY_RESOLVED, "dom-a"),
            (ResolutionStatus.TRUST_RESOLVED, "dom-a"),
            (ResolutionStatus.HUMAN_REVIEWED, "dom-a"),
            (ResolutionStatus.DETECTED, "dom-b"),
        ]:
            r = make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.8),
                    make_node_value("n2", '"v2"', 0.4),
                ],
                status=status,
                domain=domain,
                conflict_id=f"conflict-{status.value}-{domain}",
            )
            records.append(r)
        return records

    def test_goodhart_get_unresolved_excludes_authority_resolved(self):
        """get_unresolved must not return AUTHORITY_RESOLVED conflicts."""
        records = self._make_records_for_filtering()
        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        result = resolver.get_unresolved(domain="dom-a")

        for r in result:
            assert r.status in (ResolutionStatus.DETECTED, ResolutionStatus.UNRESOLVABLE)
            assert r.status != ResolutionStatus.AUTHORITY_RESOLVED

    def test_goodhart_get_unresolved_excludes_trust_resolved(self):
        """get_unresolved must not return TRUST_RESOLVED conflicts."""
        records = self._make_records_for_filtering()
        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        result = resolver.get_unresolved(domain="dom-a")

        for r in result:
            assert r.status != ResolutionStatus.TRUST_RESOLVED

    def test_goodhart_get_unresolved_excludes_human_reviewed(self):
        """get_unresolved must not return HUMAN_REVIEWED conflicts."""
        records = self._make_records_for_filtering()
        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        result = resolver.get_unresolved(domain="dom-a")

        for r in result:
            assert r.status != ResolutionStatus.HUMAN_REVIEWED

    def test_goodhart_get_unresolved_domain_filter_exact_match(self):
        """Domain filtering must use exact match, not substring or prefix matching."""
        records = [
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.8),
                    make_node_value("n2", '"v2"', 0.4),
                ],
                status=ResolutionStatus.DETECTED,
                domain="auth",
                conflict_id="c-auth",
            ),
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.8),
                    make_node_value("n2", '"v2"', 0.4),
                ],
                status=ResolutionStatus.DETECTED,
                domain="authorization",
                conflict_id="c-authorization",
            ),
        ]
        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        result = resolver.get_unresolved(domain="auth")

        assert all(r.domain == "auth" for r in result)
        assert not any(r.domain == "authorization" for r in result)

    def test_goodhart_get_unresolved_ordering_with_same_timestamp(self):
        """get_unresolved with same timestamps should return a stable result without error."""
        ts = "2024-01-01T00:00:00Z"
        records = [
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.8),
                    make_node_value("n2", '"v2"', 0.4),
                ],
                status=ResolutionStatus.DETECTED,
                domain="dom-x",
                conflict_id=f"c-{i}",
            )
            for i in range(3)
        ]
        # Set same detected_at
        for r in records:
            r.detected_at = ts

        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        result = resolver.get_unresolved(domain="dom-x")
        assert len(result) == 3
        assert all(r.status == ResolutionStatus.DETECTED for r in result)


# ─── has_blocking_conflicts() tests ───────────────────────────────────────────

class TestGoodhartHasBlocking:

    def test_goodhart_has_blocking_human_reviewed_not_blocking(self):
        """After human review clears a blocking conflict, has_blocking_conflicts must return False."""
        reviewed = make_conflict_record(
            competing_values=[
                make_node_value("n1", '"v1"', 0.5),
                make_node_value("n2", '"v2"', 0.5),
            ],
            status=ResolutionStatus.HUMAN_REVIEWED,
            domain="deploy-dom",
            data_tier="critical",
            blocks_deploy=False,  # Human review clears this
            conflict_id="c-reviewed",
        )

        store, stored = make_mock_store([reviewed])

        try:
            resolver = ConflictResolver(
                config=make_config(protected_tiers=["critical"]),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(
                make_config(protected_tiers=["critical"]),
                store,
                make_mock_signal_emitter(),
            )

        result = resolver.has_blocking_conflicts("deploy-dom")
        assert result is False

    def test_goodhart_has_blocking_detected_protected_blocks(self):
        """A DETECTED conflict in a protected tier should block deployment."""
        detected = make_conflict_record(
            competing_values=[
                make_node_value("n1", '"v1"', 0.5),
                make_node_value("n2", '"v2"', 0.5),
            ],
            status=ResolutionStatus.DETECTED,
            domain="blocking-dom",
            data_tier="critical",
            blocks_deploy=True,
            conflict_id="c-detected-blocking",
        )

        store, stored = make_mock_store([detected])

        try:
            resolver = ConflictResolver(
                config=make_config(protected_tiers=["critical"]),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(
                make_config(protected_tiers=["critical"]),
                store,
                make_mock_signal_emitter(),
            )

        result = resolver.has_blocking_conflicts("blocking-dom")
        assert result is True


# ─── get_summary() tests ──────────────────────────────────────────────────────

class TestGoodhartGetSummary:

    def test_goodhart_get_summary_deploy_blocking_excludes_resolved(self):
        """deploy_blocking_count must not count resolved conflicts even if they have blocks_deploy=True."""
        records = [
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.5),
                    make_node_value("n2", '"v2"', 0.5),
                ],
                status=ResolutionStatus.UNRESOLVABLE,
                domain="dom-a",
                blocks_deploy=True,
                conflict_id="c-unresolvable-blocking",
            ),
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.5),
                    make_node_value("n2", '"v2"', 0.5),
                ],
                status=ResolutionStatus.AUTHORITY_RESOLVED,
                domain="dom-a",
                blocks_deploy=True,  # This should NOT count
                conflict_id="c-resolved-blocking",
            ),
        ]

        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        summary = resolver.get_summary()
        assert summary.deploy_blocking_count == 1

    def test_goodhart_get_summary_domains_affected_only_unresolved(self):
        """domains_affected should only include domains with DETECTED or UNRESOLVABLE conflicts."""
        records = [
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.5),
                    make_node_value("n2", '"v2"', 0.5),
                ],
                status=ResolutionStatus.AUTHORITY_RESOLVED,
                domain="dom-resolved-only",
                conflict_id="c-resolved",
            ),
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.5),
                    make_node_value("n2", '"v2"', 0.5),
                ],
                status=ResolutionStatus.DETECTED,
                domain="dom-with-unresolved",
                conflict_id="c-detected",
            ),
        ]

        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        summary = resolver.get_summary()
        assert "dom-with-unresolved" in summary.domains_affected
        assert "dom-resolved-only" not in summary.domains_affected

    def test_goodhart_get_summary_unresolved_count_includes_both(self):
        """unresolved_count must include both DETECTED and UNRESOLVABLE."""
        records = [
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.5),
                    make_node_value("n2", '"v2"', 0.5),
                ],
                status=ResolutionStatus.DETECTED,
                domain="d",
                conflict_id=f"c-det-{i}",
            )
            for i in range(2)
        ] + [
            make_conflict_record(
                competing_values=[
                    make_node_value("n1", '"v1"', 0.5),
                    make_node_value("n2", '"v2"', 0.5),
                ],
                status=ResolutionStatus.UNRESOLVABLE,
                domain="d",
                conflict_id=f"c-unr-{i}",
            )
            for i in range(3)
        ]

        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        summary = resolver.get_summary()
        assert summary.unresolved_count == 5

    def test_goodhart_get_summary_total_equals_sum_with_all_statuses(self):
        """total_conflicts must equal the sum of all status counts, verified with non-trivial mix."""
        records = []
        status_counts = {
            ResolutionStatus.DETECTED: 2,
            ResolutionStatus.UNRESOLVABLE: 1,
            ResolutionStatus.AUTHORITY_RESOLVED: 3,
            ResolutionStatus.TRUST_RESOLVED: 2,
            ResolutionStatus.HUMAN_REVIEWED: 1,
        }
        idx = 0
        for status, count in status_counts.items():
            for _ in range(count):
                records.append(make_conflict_record(
                    competing_values=[
                        make_node_value("n1", '"v1"', 0.5),
                        make_node_value("n2", '"v2"', 0.5),
                    ],
                    status=status,
                    domain=f"d-{idx % 3}",
                    conflict_id=f"c-{idx}",
                ))
                idx += 1

        store, stored = make_mock_store(records)

        try:
            resolver = ConflictResolver(
                config=make_config(),
                store=store,
                signal_emitter=make_mock_signal_emitter(),
            )
        except TypeError:
            resolver = ConflictResolver(make_config(), store, make_mock_signal_emitter())

        summary = resolver.get_summary()

        assert summary.total_conflicts == 9
        assert summary.unresolved_count == 3  # 2 DETECTED + 1 UNRESOLVABLE
        assert summary.authority_resolved_count == 3
        assert summary.trust_resolved_count == 2
        assert summary.human_reviewed_count == 1
        total_sum = (
            summary.unresolved_count
            + summary.authority_resolved_count
            + summary.trust_resolved_count
            + summary.human_reviewed_count
        )
        assert summary.total_conflicts == total_sum


# ─── load_config() tests ──────────────────────────────────────────────────────

class TestGoodhartLoadConfig:

    def test_goodhart_load_config_authority_floor_at_1_0(self, tmp_path):
        """authority_override_floor at exactly 1.0 should be valid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 1.0\n"
            "authority_override_floor: 1.0\n"
            "trust_delta_threshold: 0.5\n"
            "checkpoint_interval: 10\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        config = load_config(str(config_file))
        assert config.authority_override_floor == 1.0

    def test_goodhart_load_config_authority_floor_at_0_0(self, tmp_path):
        """authority_override_floor at exactly 0.0 should be valid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 1.0\n"
            "authority_override_floor: 0.0\n"
            "trust_delta_threshold: 0.5\n"
            "checkpoint_interval: 10\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        config = load_config(str(config_file))
        assert config.authority_override_floor == 0.0

    def test_goodhart_load_config_threshold_slightly_above_1(self, tmp_path):
        """Threshold value of 1.001 should be rejected as invalid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 1.0\n"
            "authority_override_floor: 0.5\n"
            "trust_delta_threshold: 1.001\n"
            "checkpoint_interval: 10\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        with pytest.raises(Exception) as exc_info:
            load_config(str(config_file))
        error_str = str(exc_info.value).lower()
        assert "config" in error_str or "valid" in error_str or "threshold" in error_str

    def test_goodhart_load_config_negative_threshold(self, tmp_path):
        """Negative threshold value should be rejected as invalid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 1.0\n"
            "authority_override_floor: -0.01\n"
            "trust_delta_threshold: 0.5\n"
            "checkpoint_interval: 10\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        with pytest.raises(Exception):
            load_config(str(config_file))

    def test_goodhart_load_config_window_timeout_at_0_1(self, tmp_path):
        """window_timeout_seconds at exactly 0.1 should be valid (minimum allowed)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 0.1\n"
            "authority_override_floor: 0.5\n"
            "trust_delta_threshold: 0.5\n"
            "checkpoint_interval: 10\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        config = load_config(str(config_file))
        assert config.window_timeout_seconds == pytest.approx(0.1)

    def test_goodhart_load_config_window_timeout_0_09_rejected(self, tmp_path):
        """window_timeout_seconds at 0.09 should be rejected (below 0.1 minimum)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 0.09\n"
            "authority_override_floor: 0.5\n"
            "trust_delta_threshold: 0.5\n"
            "checkpoint_interval: 10\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        with pytest.raises(Exception):
            load_config(str(config_file))

    def test_goodhart_load_config_checkpoint_interval_exactly_1(self, tmp_path):
        """checkpoint_interval of exactly 1 should be valid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "window_timeout_seconds: 1.0\n"
            "authority_override_floor: 0.5\n"
            "trust_delta_threshold: 0.5\n"
            "checkpoint_interval: 1\n"
            "conflict_log_path: /tmp/log.jsonl\n"
            "protected_tiers:\n  - critical\n"
        )

        config = load_config(str(config_file))
        assert config.checkpoint_interval == 1
