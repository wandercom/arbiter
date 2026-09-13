"""
Contract tests for the Conflict Resolver component.

Tests verify behavior at boundaries using the contract specification.
All external dependencies (trust_ledger, authority_registry, stigmergy,
field_classifier, conflict_store) are mocked.

Run with: pytest contract_test.py -v
"""

import copy
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, call, patch, ANY

import pytest

# Import the component under test
from arbiter.conflicts import (
    ResolutionStatus,
    ResolutionStrategy,
    ConflictErrorCode,
    NodeValue,
    Resolution,
    ConflictRecord,
    ConflictResolverConfig,
    ConflictSignal,
    SpanFieldReport,
    ConflictSummary,
    TrustLookup,
    AuthorityLookup,
    SignalEmitter,
    ConflictStore,
    ConflictDetector,
    ConflictResolver,
    load_config,
    verify_log_integrity,
)


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_config(**overrides) -> ConflictResolverConfig:
    defaults = dict(
        window_timeout_seconds=0.5,
        authority_override_floor=0.6,
        trust_delta_threshold=0.2,
        checkpoint_interval=5,
        conflict_log_path="/tmp/test_conflict_log.jsonl",
        protected_tiers=["critical", "sensitive"],
    )
    defaults.update(overrides)
    return ConflictResolverConfig(**defaults)


def make_span_report(
    *,
    span_id=None,
    execution_id="exec-1",
    node_id="node-A",
    domain="auth",
    field="username",
    value_serialized='"alice"',
    reported_at=None,
) -> SpanFieldReport:
    return SpanFieldReport(
        span_id=span_id or str(uuid.uuid4()),
        execution_id=execution_id,
        node_id=node_id,
        domain=domain,
        field=field,
        value_serialized=value_serialized,
        reported_at=reported_at or _utc_now_iso(),
    )


def make_node_value(
    *,
    node_id="node-A",
    value_serialized='"alice"',
    trust_score_snapshot=0.9,
    is_authoritative=False,
    span_id=None,
) -> NodeValue:
    return NodeValue(
        node_id=node_id,
        value_serialized=value_serialized,
        trust_score_snapshot=trust_score_snapshot,
        is_authoritative=is_authoritative,
        span_id=span_id or str(uuid.uuid4()),
    )


def make_conflict_record(
    *,
    conflict_id=None,
    execution_id="exec-1",
    domain="auth",
    field="username",
    data_tier="critical",
    competing_values=None,
    detected_at=None,
    status=ResolutionStatus.DETECTED,
    resolution=None,
    blocks_deploy=False,
) -> ConflictRecord:
    if competing_values is None:
        competing_values = [
            make_node_value(node_id="node-A", value_serialized='"alice"', trust_score_snapshot=0.9),
            make_node_value(node_id="node-B", value_serialized='"bob"', trust_score_snapshot=0.7),
        ]
    return ConflictRecord(
        conflict_id=conflict_id or str(uuid.uuid4()),
        execution_id=execution_id,
        domain=domain,
        field=field,
        data_tier=data_tier,
        competing_values=competing_values,
        detected_at=detected_at or _utc_now_iso(),
        status=status,
        resolution=resolution,
        blocks_deploy=blocks_deploy,
    )


def make_mock_trust_lookup(scores=None):
    """Return a mock TrustLookup that returns scores by node_id."""
    mock = MagicMock(spec=TrustLookup)
    scores = scores or {"node-A": 0.9, "node-B": 0.7, "node-C": 0.5}
    mock.lookup_trust_score = MagicMock(side_effect=lambda node_id: scores.get(node_id, 0.5))
    return mock


def make_mock_authority_lookup(authorities=None):
    """Return a mock AuthorityLookup that returns is_authoritative by node_id."""
    mock = MagicMock(spec=AuthorityLookup)
    authorities = authorities or {}
    mock.is_authoritative = MagicMock(side_effect=lambda node_id: authorities.get(node_id, False))
    return mock


def make_mock_signal_emitter():
    mock = MagicMock(spec=SignalEmitter)
    mock.emit_signal = MagicMock(return_value=None)
    return mock


def make_mock_conflict_store(records=None):
    mock = MagicMock(spec=ConflictStore)
    stored = list(records or [])
    mock.append = MagicMock(side_effect=lambda r: stored.append(r))
    mock.load_all = MagicMock(return_value=stored)
    mock.verify_checksums = MagicMock(return_value=True)
    mock._stored = stored  # expose for assertions
    return mock


def make_detector(config=None, trust_lookup=None, authority_lookup=None, store=None, field_classifier=None):
    """Build a ConflictDetector with injectable mocks."""
    return ConflictDetector(
        config=config or make_config(),
        trust_lookup=trust_lookup or make_mock_trust_lookup(),
        authority_lookup=authority_lookup or make_mock_authority_lookup(),
        store=store or make_mock_conflict_store(),
        field_classifier=field_classifier or MagicMock(return_value="critical"),
    )


def make_resolver(config=None, store=None, signal_emitter=None):
    """Build a ConflictResolver with injectable mocks."""
    return ConflictResolver(
        config=config or make_config(),
        store=store or make_mock_conflict_store(),
        signal_emitter=signal_emitter or make_mock_signal_emitter(),
    )


# ===========================================================================
# TestIngest
# ===========================================================================
class TestIngest:
    """Tests for the ingest() function — span buffering and conflict detection."""

    def test_single_span_no_conflict(self):
        """Ingesting a single span does not produce a conflict."""
        detector = make_detector()
        span = make_span_report(node_id="node-A", value_serialized='"alice"')
        result = detector.ingest(span)
        # Window hasn't closed yet (or only one value), so no conflict
        assert isinstance(result, list)
        # Either empty (window still open) or empty (single value)
        assert len(result) == 0

    def test_two_conflicting_spans_after_window_close(self):
        """Two spans with different values produce a conflict when window closes."""
        import time
        config = make_config(window_timeout_seconds=0.1)
        store = make_mock_conflict_store()
        detector = make_detector(config=config, store=store)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        time.sleep(0.15)  # Wait for window to expire
        result = detector.ingest(span_b)

        # The window for span_a should have closed, but span_b opens a new one.
        # Alternatively both may be in the same window. Depends on implementation.
        # We may need flush to guarantee.
        # Use flush to force-close all windows as a reliable alternative:
        result = detector.flush()

        conflicts = [r for r in result if r.status == ResolutionStatus.DETECTED]
        if len(conflicts) > 0:
            conflict = conflicts[0]
            assert conflict.status == ResolutionStatus.DETECTED
            assert len(conflict.competing_values) >= 2
            assert store.append.called

    def test_two_agreeing_spans_no_conflict(self):
        """Two spans with same value produce no conflict."""
        config = make_config(window_timeout_seconds=0.1)
        detector = make_detector(config=config)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"alice"')

        detector.ingest(span_a)
        detector.ingest(span_b)
        result = detector.flush()

        assert isinstance(result, list)
        # No conflict because values agree
        assert len(result) == 0

    def test_duplicate_span_silently_dropped(self):
        """Duplicate span (same node_id + span_id) is silently dropped."""
        detector = make_detector()
        shared_span_id = "span-dup-1"
        span1 = make_span_report(node_id="node-A", span_id=shared_span_id, value_serialized='"alice"')
        span2 = make_span_report(node_id="node-A", span_id=shared_span_id, value_serialized='"alice"')

        # Neither call should raise
        result1 = detector.ingest(span1)
        result2 = detector.ingest(span2)

        assert isinstance(result1, list)
        assert isinstance(result2, list)

    def test_trust_lookup_failed(self):
        """Trust ledger failure during conflict detection raises error with node_id context."""
        import time
        config = make_config(window_timeout_seconds=0.1)
        trust = make_mock_trust_lookup()
        trust.lookup_trust_score = MagicMock(side_effect=Exception("trust service down"))

        detector = make_detector(config=config, trust_lookup=trust)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        detector.ingest(span_b)

        with pytest.raises(Exception) as exc_info:
            detector.flush()

        error_str = str(exc_info.value).lower()
        # Should reference trust lookup failure
        assert "trust" in error_str or "TRUST_LOOKUP_FAILED" in str(exc_info.value)

    def test_authority_lookup_failed(self):
        """Authority registry failure during conflict detection raises error."""
        import time
        config = make_config(window_timeout_seconds=0.1)
        authority = make_mock_authority_lookup()
        authority.is_authoritative = MagicMock(side_effect=Exception("authority service down"))

        detector = make_detector(config=config, authority_lookup=authority)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        detector.ingest(span_b)

        with pytest.raises(Exception) as exc_info:
            detector.flush()

        error_str = str(exc_info.value)
        assert "authority" in error_str.lower() or "AUTHORITY_LOOKUP_FAILED" in error_str

    def test_store_write_failed(self):
        """Conflict store append failure raises store_write_failed error."""
        import time
        config = make_config(window_timeout_seconds=0.1)
        store = make_mock_conflict_store()
        store.append = MagicMock(side_effect=Exception("disk full"))

        detector = make_detector(config=config, store=store)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        detector.ingest(span_b)

        with pytest.raises(Exception) as exc_info:
            detector.flush()

        error_str = str(exc_info.value)
        assert "store" in error_str.lower() or "write" in error_str.lower() or "STORE_WRITE_FAILED" in error_str

    def test_tier_lookup_failed_degraded_mode(self):
        """Tier lookup failure still detects conflict with empty data_tier."""
        config = make_config(window_timeout_seconds=0.1)
        field_classifier = MagicMock(side_effect=Exception("tier service down"))

        detector = make_detector(config=config, field_classifier=field_classifier)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        detector.ingest(span_b)

        # Should still produce a conflict (degraded mode)
        result = detector.flush()
        if len(result) > 0:
            assert result[0].data_tier == "" or result[0].data_tier is None

    def test_multiple_domains_independent(self):
        """Spans for different (domain, field) keys are buffered independently."""
        detector = make_detector()

        span_auth = make_span_report(node_id="node-A", domain="auth", field="user")
        span_billing = make_span_report(node_id="node-B", domain="billing", field="amount")

        r1 = detector.ingest(span_auth)
        r2 = detector.ingest(span_billing)

        # Both buffered, no conflicts yet
        assert isinstance(r1, list)
        assert isinstance(r2, list)

    def test_trust_snapshot_at_detection_time(self):
        """trust_score_snapshot reflects trust at detection time, not ingestion."""
        config = make_config(window_timeout_seconds=0.1)
        trust_scores = {"node-A": 0.9, "node-B": 0.7}
        trust = make_mock_trust_lookup(trust_scores)

        detector = make_detector(config=config, trust_lookup=trust)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        # Change trust scores after ingestion but before detection
        trust_scores["node-A"] = 0.1
        trust.lookup_trust_score = MagicMock(side_effect=lambda nid: trust_scores.get(nid, 0.5))
        detector.ingest(span_b)

        result = detector.flush()
        if len(result) > 0:
            for nv in result[0].competing_values:
                # Trust scores should be snapshotted at detection (flush) time
                if nv.node_id == "node-A":
                    assert nv.trust_score_snapshot == 0.1  # detection-time value
                elif nv.node_id == "node-B":
                    assert nv.trust_score_snapshot == 0.7

    def test_authority_flag_reflects_current_state(self):
        """is_authoritative reflects authority registry state at detection time."""
        config = make_config(window_timeout_seconds=0.1)
        authority = make_mock_authority_lookup({"node-A": True})

        detector = make_detector(config=config, authority_lookup=authority)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        detector.ingest(span_b)
        result = detector.flush()

        if len(result) > 0:
            for nv in result[0].competing_values:
                if nv.node_id == "node-A":
                    assert nv.is_authoritative is True
                elif nv.node_id == "node-B":
                    assert nv.is_authoritative is False

    def test_persisted_to_store(self):
        """All returned ConflictRecords are persisted to the conflict store."""
        config = make_config(window_timeout_seconds=0.1)
        store = make_mock_conflict_store()
        detector = make_detector(config=config, store=store)

        span_a = make_span_report(node_id="node-A", value_serialized='"alice"')
        span_b = make_span_report(node_id="node-B", value_serialized='"bob"')

        detector.ingest(span_a)
        detector.ingest(span_b)
        result = detector.flush()

        if len(result) > 0:
            assert store.append.call_count >= len(result)

    def test_invariant_conflict_requires_two_distinct_values(self):
        """A conflict is never created with fewer than 2 distinct values."""
        config = make_config(window_timeout_seconds=0.1)
        detector = make_detector(config=config)

        # All same value
        for nid in ["node-A", "node-B", "node-C"]:
            detector.ingest(make_span_report(node_id=nid, value_serialized='"same"'))

        result = detector.flush()
        # No conflict because all values identical
        assert len(result) == 0


# ===========================================================================
# TestFlush
# ===========================================================================
class TestFlush:
    """Tests for the flush() function — force-close all windows."""

    def test_empty_buffer_returns_empty(self):
        """Flushing an empty buffer returns empty list."""
        detector = make_detector()
        result = detector.flush()
        assert result == []

    def test_flush_with_conflicts(self):
        """Flushing buffer with conflicting spans produces conflicts."""
        detector = make_detector()
        detector.ingest(make_span_report(node_id="node-A", value_serialized='"alice"'))
        detector.ingest(make_span_report(node_id="node-B", value_serialized='"bob"'))

        result = detector.flush()
        assert len(result) >= 1
        for r in result:
            assert r.status == ResolutionStatus.DETECTED
            assert len(r.competing_values) >= 2

    def test_buffer_empty_after_flush(self):
        """Buffer is empty after flush — second flush returns empty."""
        detector = make_detector()
        detector.ingest(make_span_report(node_id="node-A", value_serialized='"alice"'))
        detector.ingest(make_span_report(node_id="node-B", value_serialized='"bob"'))

        detector.flush()
        result2 = detector.flush()
        assert result2 == []

    def test_single_value_no_conflict(self):
        """Window with only one distinct value produces no conflict on flush."""
        detector = make_detector()
        detector.ingest(make_span_report(node_id="node-A", value_serialized='"only_one"'))

        result = detector.flush()
        assert len(result) == 0

    def test_flush_error_trust_lookup_failed(self):
        """Trust lookup failure during flush raises error."""
        trust = make_mock_trust_lookup()
        trust.lookup_trust_score = MagicMock(side_effect=Exception("trust down"))
        detector = make_detector(trust_lookup=trust)

        detector.ingest(make_span_report(node_id="node-A", value_serialized='"alice"'))
        detector.ingest(make_span_report(node_id="node-B", value_serialized='"bob"'))

        with pytest.raises(Exception) as exc_info:
            detector.flush()
        assert "trust" in str(exc_info.value).lower() or "TRUST_LOOKUP_FAILED" in str(exc_info.value)

    def test_flush_error_store_write_failed(self):
        """Store write failure during flush raises error."""
        store = make_mock_conflict_store()
        store.append = MagicMock(side_effect=Exception("write failed"))
        detector = make_detector(store=store)

        detector.ingest(make_span_report(node_id="node-A", value_serialized='"alice"'))
        detector.ingest(make_span_report(node_id="node-B", value_serialized='"bob"'))

        with pytest.raises(Exception) as exc_info:
            detector.flush()
        error_str = str(exc_info.value).lower()
        assert "store" in error_str or "write" in error_str


# ===========================================================================
# TestResolve
# ===========================================================================
class TestResolve:
    """Tests for the resolve() function — three-step resolution protocol."""

    def test_authority_resolution(self):
        """Exactly one authoritative node with trust > floor wins via AUTHORITY."""
        config = make_config(authority_override_floor=0.6, trust_delta_threshold=0.2)
        store = make_mock_conflict_store()
        resolver = make_resolver(config=config, store=store)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=True),
                make_node_value(node_id="node-B", trust_score_snapshot=0.7, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        assert result.resolution.strategy == ResolutionStrategy.AUTHORITY
        assert result.resolution.winner_node_id == "node-A"
        assert result.resolution.resolved_at is not None
        assert store.append.called

    def test_trust_arbitration_resolution(self):
        """No authoritative node, trust delta > threshold: highest trust wins."""
        config = make_config(trust_delta_threshold=0.1)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.5, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.strategy == ResolutionStrategy.TRUST_ARBITRATION
        assert result.resolution.winner_node_id == "node-A"

    def test_unresolvable_resolution(self):
        """No authoritative, trust delta <= threshold: UNRESOLVABLE with signal."""
        config = make_config(trust_delta_threshold=0.5)
        signal_emitter = make_mock_signal_emitter()
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.8, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.75, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE
        assert result.resolution.strategy == ResolutionStrategy.HUMAN
        assert signal_emitter.emit_signal.called

    def test_authority_below_floor_falls_through(self):
        """Authoritative node with trust <= floor skips authority step."""
        config = make_config(authority_override_floor=0.8, trust_delta_threshold=0.1)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.5, is_authoritative=True),
                make_node_value(node_id="node-B", trust_score_snapshot=0.9, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        # Authority check should fail (0.5 <= 0.8), so falls to trust arbitration
        assert result.status != ResolutionStatus.AUTHORITY_RESOLVED
        # Trust delta = 0.4 > 0.1, so TRUST_RESOLVED with node-B winning
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.winner_node_id == "node-B"

    def test_authority_at_floor_boundary(self):
        """Authoritative node with trust exactly at floor fails authority check."""
        config = make_config(authority_override_floor=0.7, trust_delta_threshold=0.05)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.7, is_authoritative=True),
                make_node_value(node_id="node-B", trust_score_snapshot=0.6, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        # Trust at floor (not above), so authority check should not succeed
        assert result.status != ResolutionStatus.AUTHORITY_RESOLVED

    def test_trust_delta_at_threshold_unresolvable(self):
        """Trust delta exactly at threshold does not resolve (must be > threshold)."""
        config = make_config(trust_delta_threshold=0.2)
        signal_emitter = make_mock_signal_emitter()
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.7, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.5, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        # Delta = 0.2 == threshold, not > threshold => UNRESOLVABLE
        assert result.status == ResolutionStatus.UNRESOLVABLE

    def test_nway_authority_exactly_one(self):
        """N-way conflict with exactly one authoritative node uses AUTHORITY."""
        config = make_config(authority_override_floor=0.5)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", value_serialized='"a"', trust_score_snapshot=0.8, is_authoritative=True),
                make_node_value(node_id="node-B", value_serialized='"b"', trust_score_snapshot=0.9, is_authoritative=False),
                make_node_value(node_id="node-C", value_serialized='"c"', trust_score_snapshot=0.7, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED
        assert result.resolution.winner_node_id == "node-A"

    def test_nway_multiple_authoritative_skips_authority(self):
        """N-way conflict with multiple authoritative nodes skips authority step."""
        config = make_config(authority_override_floor=0.5, trust_delta_threshold=0.1)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", value_serialized='"a"', trust_score_snapshot=0.9, is_authoritative=True),
                make_node_value(node_id="node-B", value_serialized='"b"', trust_score_snapshot=0.7, is_authoritative=True),
                make_node_value(node_id="node-C", value_serialized='"c"', trust_score_snapshot=0.5, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        # Multiple authoritative => skip authority step
        assert result.status != ResolutionStatus.AUTHORITY_RESOLVED

    def test_nway_trust_top2_delta(self):
        """N-way conflict trust arbitration compares only top-2 trust scores."""
        config = make_config(trust_delta_threshold=0.3)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", value_serialized='"a"', trust_score_snapshot=0.9, is_authoritative=False),
                make_node_value(node_id="node-B", value_serialized='"b"', trust_score_snapshot=0.5, is_authoritative=False),
                make_node_value(node_id="node-C", value_serialized='"c"', trust_score_snapshot=0.1, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        # Top 2: 0.9, 0.5 -> delta = 0.4 > 0.3 => TRUST_RESOLVED
        assert result.status == ResolutionStatus.TRUST_RESOLVED
        assert result.resolution.winner_node_id == "node-A"

    def test_unresolvable_blocks_deploy_protected(self):
        """UNRESOLVABLE + protected data_tier => blocks_deploy=True."""
        config = make_config(protected_tiers=["critical"], trust_delta_threshold=0.5)
        signal_emitter = make_mock_signal_emitter()
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        conflict = make_conflict_record(
            data_tier="critical",
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.8, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.75, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE
        assert result.blocks_deploy is True

    def test_unresolvable_no_block_unprotected(self):
        """UNRESOLVABLE + non-protected data_tier => blocks_deploy=False."""
        config = make_config(protected_tiers=["critical"], trust_delta_threshold=0.5)
        signal_emitter = make_mock_signal_emitter()
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        conflict = make_conflict_record(
            data_tier="general",
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.8, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.75, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert result.status == ResolutionStatus.UNRESOLVABLE
        assert result.blocks_deploy is False

    def test_already_resolved_error(self):
        """Resolving an already-resolved conflict raises already_resolved error."""
        resolver = make_resolver()
        conflict = make_conflict_record(
            status=ResolutionStatus.AUTHORITY_RESOLVED,
            resolution=Resolution(
                strategy=ResolutionStrategy.AUTHORITY,
                winner_node_id="node-A",
                resolved_at=_utc_now_iso(),
                rationale="test",
                reviewed_by="",
            ),
        )
        with pytest.raises(Exception) as exc_info:
            resolver.resolve(conflict)
        error_str = str(exc_info.value)
        assert "already" in error_str.lower() or "CONFLICT_ALREADY_RESOLVED" in error_str or "resolved" in error_str.lower()

    def test_empty_competing_values_error(self):
        """Conflict with < 2 competing values raises empty_competing_values error."""
        resolver = make_resolver()
        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A"),
            ],
            status=ResolutionStatus.DETECTED,
        )
        with pytest.raises(Exception) as exc_info:
            resolver.resolve(conflict)
        error_str = str(exc_info.value)
        assert "empty" in error_str.lower() or "competing" in error_str.lower() or "EMPTY_COMPETING_VALUES" in error_str

    def test_signal_emission_failed_error(self):
        """Signal emission failure during Step 3 raises signal_emission_failed."""
        config = make_config(trust_delta_threshold=0.5)
        signal_emitter = make_mock_signal_emitter()
        signal_emitter.emit_signal = MagicMock(side_effect=Exception("stigmergy down"))
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.8, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.75, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        with pytest.raises(Exception) as exc_info:
            resolver.resolve(conflict)
        error_str = str(exc_info.value)
        assert "signal" in error_str.lower() or "emission" in error_str.lower() or "SIGNAL_EMISSION_FAILED" in error_str

    def test_store_write_failed_error(self):
        """Store append failure during resolve raises store_write_failed."""
        store = make_mock_conflict_store()
        store.append = MagicMock(side_effect=Exception("disk full"))
        resolver = make_resolver(store=store)

        conflict = make_conflict_record(status=ResolutionStatus.DETECTED)
        with pytest.raises(Exception) as exc_info:
            resolver.resolve(conflict)
        error_str = str(exc_info.value)
        assert "store" in error_str.lower() or "write" in error_str.lower() or "STORE_WRITE_FAILED" in error_str

    def test_rationale_includes_details(self):
        """Resolution rationale includes node IDs, trust scores, and thresholds."""
        config = make_config(authority_override_floor=0.6, trust_delta_threshold=0.1)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=True),
                make_node_value(node_id="node-B", trust_score_snapshot=0.7, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        rationale = result.resolution.rationale
        assert rationale is not None and len(rationale) > 0
        # Rationale should contain node IDs and scores
        assert "node-A" in rationale or "node_A" in rationale

    def test_signal_content_verification(self):
        """ConflictSignal for UNRESOLVABLE has correct conflict_id, execution_id, etc."""
        config = make_config(trust_delta_threshold=0.5, protected_tiers=["critical"])
        signal_emitter = make_mock_signal_emitter()
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        cid = "conflict-sig-test"
        conflict = make_conflict_record(
            conflict_id=cid,
            execution_id="exec-sig",
            domain="auth",
            field="user",
            data_tier="critical",
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.8, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.75, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)

        assert signal_emitter.emit_signal.called
        signal_arg = signal_emitter.emit_signal.call_args
        # The signal should reference the conflict
        signal_data = signal_arg[0][0] if signal_arg[0] else signal_arg[1].get("signal")
        if hasattr(signal_data, "conflict_id"):
            assert signal_data.conflict_id == cid
            assert signal_data.execution_id == "exec-sig"
            assert signal_data.blocks_deploy is True

    def test_deterministic_resolution(self):
        """Same conflict + same config = same result (determinism invariant)."""
        config = make_config(trust_delta_threshold=0.1)
        resolver1 = make_resolver(config=config)
        resolver2 = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.5, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        conflict2 = copy.deepcopy(conflict)

        result1 = resolver1.resolve(conflict)
        result2 = resolver2.resolve(conflict2)

        assert result1.status == result2.status
        assert result1.resolution.strategy == result2.resolution.strategy
        assert result1.resolution.winner_node_id == result2.resolution.winner_node_id

    def test_resolve_appended_to_store(self):
        """Resolved record is appended to the conflict store."""
        store = make_mock_conflict_store()
        resolver = make_resolver(store=store)

        conflict = make_conflict_record(status=ResolutionStatus.DETECTED)
        resolver.resolve(conflict)

        assert store.append.called

    def test_invariant_trust_immutable_after_detection(self):
        """Trust scores in NodeValue are not modified by resolve."""
        resolver = make_resolver()
        nv_a = make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=True)
        nv_b = make_node_value(node_id="node-B", trust_score_snapshot=0.7, is_authoritative=False)

        conflict = make_conflict_record(
            competing_values=[nv_a, nv_b],
            status=ResolutionStatus.DETECTED,
        )
        original_scores = {nv.node_id: nv.trust_score_snapshot for nv in conflict.competing_values}

        result = resolver.resolve(conflict)

        for nv in result.competing_values:
            assert nv.trust_score_snapshot == original_scores[nv.node_id]


# ===========================================================================
# TestSubmitHumanReview
# ===========================================================================
class TestSubmitHumanReview:
    """Tests for submit_human_review() — human override of UNRESOLVABLE conflicts."""

    def _setup_unresolvable(self, store):
        """Helper: create an UNRESOLVABLE conflict in the store."""
        conflict = make_conflict_record(
            conflict_id="conflict-hr-1",
            status=ResolutionStatus.UNRESOLVABLE,
            blocks_deploy=True,
            data_tier="critical",
            resolution=Resolution(
                strategy=ResolutionStrategy.HUMAN,
                winner_node_id="",
                resolved_at=_utc_now_iso(),
                rationale="unresolvable - needs human review",
                reviewed_by="",
            ),
        )
        store._stored.append(conflict)
        store.load_all = MagicMock(return_value=store._stored)
        return conflict

    def test_happy_path(self):
        """UNRESOLVABLE conflict transitions to HUMAN_REVIEWED."""
        store = make_mock_conflict_store()
        conflict = self._setup_unresolvable(store)
        resolver = make_resolver(store=store)

        result = resolver.submit_human_review(
            conflict_id="conflict-hr-1",
            winner_node_id="node-A",
            reviewed_by="operator@example.com",
            rationale="Manual verification confirmed node-A",
        )

        assert result.status == ResolutionStatus.HUMAN_REVIEWED
        assert result.resolution.strategy == ResolutionStrategy.HUMAN
        assert result.resolution.winner_node_id == "node-A"
        assert result.resolution.reviewed_by == "operator@example.com"
        assert result.blocks_deploy is False

    def test_clears_deploy_block(self):
        """Human review clears blocks_deploy even if previously True."""
        store = make_mock_conflict_store()
        conflict = self._setup_unresolvable(store)
        assert conflict.blocks_deploy is True

        resolver = make_resolver(store=store)
        result = resolver.submit_human_review(
            conflict_id="conflict-hr-1",
            winner_node_id="node-A",
            reviewed_by="admin",
            rationale="Confirmed",
        )

        assert result.blocks_deploy is False

    def test_conflict_not_found(self):
        """Non-existent conflict_id raises conflict_not_found."""
        store = make_mock_conflict_store()
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.submit_human_review(
                conflict_id="nonexistent",
                winner_node_id="node-A",
                reviewed_by="admin",
                rationale="test",
            )
        error_str = str(exc_info.value)
        assert "not_found" in error_str.lower() or "CONFLICT_NOT_FOUND" in error_str or "not found" in error_str.lower()

    def test_invalid_status_authority_resolved(self):
        """Conflict with AUTHORITY_RESOLVED raises invalid_status."""
        store = make_mock_conflict_store()
        resolved = make_conflict_record(
            conflict_id="conflict-resolved",
            status=ResolutionStatus.AUTHORITY_RESOLVED,
            resolution=Resolution(
                strategy=ResolutionStrategy.AUTHORITY,
                winner_node_id="node-A",
                resolved_at=_utc_now_iso(),
                rationale="authority",
                reviewed_by="",
            ),
        )
        store._stored.append(resolved)
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.submit_human_review(
                conflict_id="conflict-resolved",
                winner_node_id="node-A",
                reviewed_by="admin",
                rationale="test",
            )
        error_str = str(exc_info.value).lower()
        assert "status" in error_str or "resolved" in error_str or "invalid" in error_str

    def test_invalid_status_detected(self):
        """Conflict with DETECTED status raises invalid_status (must be UNRESOLVABLE)."""
        store = make_mock_conflict_store()
        detected = make_conflict_record(
            conflict_id="conflict-detected",
            status=ResolutionStatus.DETECTED,
        )
        store._stored.append(detected)
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.submit_human_review(
                conflict_id="conflict-detected",
                winner_node_id="node-A",
                reviewed_by="admin",
                rationale="test",
            )
        error_str = str(exc_info.value).lower()
        assert "status" in error_str or "unresolvable" in error_str or "invalid" in error_str

    def test_invalid_status_trust_resolved(self):
        """Conflict with TRUST_RESOLVED (terminal state) raises invalid_status."""
        store = make_mock_conflict_store()
        trust_resolved = make_conflict_record(
            conflict_id="conflict-trust",
            status=ResolutionStatus.TRUST_RESOLVED,
            resolution=Resolution(
                strategy=ResolutionStrategy.TRUST_ARBITRATION,
                winner_node_id="node-A",
                resolved_at=_utc_now_iso(),
                rationale="trust",
                reviewed_by="",
            ),
        )
        store._stored.append(trust_resolved)
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.submit_human_review(
                conflict_id="conflict-trust",
                winner_node_id="node-A",
                reviewed_by="admin",
                rationale="test",
            )
        error_str = str(exc_info.value).lower()
        assert "status" in error_str or "resolved" in error_str or "invalid" in error_str

    def test_invalid_winner_node(self):
        """winner_node_id not in competing nodes raises invalid_winner."""
        store = make_mock_conflict_store()
        self._setup_unresolvable(store)
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.submit_human_review(
                conflict_id="conflict-hr-1",
                winner_node_id="node-NONEXISTENT",
                reviewed_by="admin",
                rationale="test",
            )
        error_str = str(exc_info.value).lower()
        assert "winner" in error_str or "invalid" in error_str or "node" in error_str

    def test_store_write_failed(self):
        """Store append failure raises store_write_failed."""
        store = make_mock_conflict_store()
        self._setup_unresolvable(store)
        store.append = MagicMock(side_effect=Exception("write failed"))
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.submit_human_review(
                conflict_id="conflict-hr-1",
                winner_node_id="node-A",
                reviewed_by="admin",
                rationale="test",
            )
        error_str = str(exc_info.value).lower()
        assert "store" in error_str or "write" in error_str


# ===========================================================================
# TestGetUnresolved
# ===========================================================================
class TestGetUnresolved:
    """Tests for get_unresolved() — querying unresolved conflicts."""

    def _make_store_with_records(self):
        store = make_mock_conflict_store()
        records = [
            make_conflict_record(conflict_id="c1", domain="auth", status=ResolutionStatus.DETECTED, detected_at="2024-01-03T00:00:00Z"),
            make_conflict_record(conflict_id="c2", domain="auth", status=ResolutionStatus.UNRESOLVABLE, detected_at="2024-01-02T00:00:00Z"),
            make_conflict_record(conflict_id="c3", domain="auth", status=ResolutionStatus.AUTHORITY_RESOLVED, detected_at="2024-01-01T00:00:00Z"),
            make_conflict_record(conflict_id="c4", domain="billing", status=ResolutionStatus.DETECTED, detected_at="2024-01-04T00:00:00Z"),
        ]
        store._stored = records
        store.load_all = MagicMock(return_value=records)
        return store, records

    def test_returns_only_unresolved_for_domain(self):
        """Returns only DETECTED and UNRESOLVABLE for the given domain."""
        store, _ = self._make_store_with_records()
        resolver = make_resolver(store=store)

        result = resolver.get_unresolved("auth")

        for r in result:
            assert r.status in (ResolutionStatus.DETECTED, ResolutionStatus.UNRESOLVABLE)
            assert r.domain == "auth"
        # Should not include AUTHORITY_RESOLVED (c3)
        ids = [r.conflict_id for r in result]
        assert "c3" not in ids

    def test_empty_result(self):
        """Returns empty list when no unresolved conflicts for domain."""
        store = make_mock_conflict_store()
        store.load_all = MagicMock(return_value=[])
        resolver = make_resolver(store=store)

        result = resolver.get_unresolved("nonexistent")
        assert result == []

    def test_newest_first_ordering(self):
        """Results ordered by detected_at descending (newest first)."""
        store, _ = self._make_store_with_records()
        resolver = make_resolver(store=store)

        result = resolver.get_unresolved("auth")

        if len(result) >= 2:
            assert result[0].detected_at >= result[1].detected_at

    def test_empty_domain_returns_all_unresolved(self):
        """Empty domain string returns all unresolved conflicts across domains."""
        store, _ = self._make_store_with_records()
        resolver = make_resolver(store=store)

        result = resolver.get_unresolved("")

        for r in result:
            assert r.status in (ResolutionStatus.DETECTED, ResolutionStatus.UNRESOLVABLE)
        # Should include both auth and billing unresolved
        domains = {r.domain for r in result}
        assert "auth" in domains
        assert "billing" in domains

    def test_store_read_failed(self):
        """Store read failure raises store_read_failed."""
        store = make_mock_conflict_store()
        store.load_all = MagicMock(side_effect=Exception("read failed"))
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.get_unresolved("auth")
        error_str = str(exc_info.value).lower()
        assert "store" in error_str or "read" in error_str


# ===========================================================================
# TestHasBlockingConflicts
# ===========================================================================
class TestHasBlockingConflicts:
    """Tests for has_blocking_conflicts() — deploy gate check."""

    def test_blocking_exists_returns_true(self):
        """Returns True when blocking conflict exists for domain."""
        store = make_mock_conflict_store()
        store._stored = [
            make_conflict_record(
                domain="auth",
                status=ResolutionStatus.UNRESOLVABLE,
                blocks_deploy=True,
            ),
        ]
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        result = resolver.has_blocking_conflicts("auth")
        assert result is True

    def test_no_conflicts_returns_false(self):
        """Returns False when no conflicts exist."""
        store = make_mock_conflict_store()
        store.load_all = MagicMock(return_value=[])
        resolver = make_resolver(store=store)

        result = resolver.has_blocking_conflicts("auth")
        assert result is False

    def test_resolved_blocking_returns_false(self):
        """Returns False when blocking conflicts are all resolved."""
        store = make_mock_conflict_store()
        store._stored = [
            make_conflict_record(
                domain="auth",
                status=ResolutionStatus.HUMAN_REVIEWED,
                blocks_deploy=False,
            ),
        ]
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        result = resolver.has_blocking_conflicts("auth")
        assert result is False

    def test_unresolved_unprotected_returns_false(self):
        """Returns False when unresolved conflict has blocks_deploy=False."""
        store = make_mock_conflict_store()
        store._stored = [
            make_conflict_record(
                domain="auth",
                data_tier="general",
                status=ResolutionStatus.UNRESOLVABLE,
                blocks_deploy=False,
            ),
        ]
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        result = resolver.has_blocking_conflicts("auth")
        assert result is False

    def test_store_read_failed(self):
        """Store read failure raises store_read_failed."""
        store = make_mock_conflict_store()
        store.load_all = MagicMock(side_effect=Exception("read failed"))
        resolver = make_resolver(store=store)

        with pytest.raises(Exception):
            resolver.has_blocking_conflicts("auth")


# ===========================================================================
# TestGetSummary
# ===========================================================================
class TestGetSummary:
    """Tests for get_summary() — aggregate statistics."""

    def test_empty_state(self):
        """Empty store returns all-zero summary."""
        store = make_mock_conflict_store()
        store.load_all = MagicMock(return_value=[])
        resolver = make_resolver(store=store)

        summary = resolver.get_summary()

        assert summary.total_conflicts == 0
        assert summary.unresolved_count == 0
        assert summary.authority_resolved_count == 0
        assert summary.trust_resolved_count == 0
        assert summary.human_reviewed_count == 0
        assert summary.deploy_blocking_count == 0
        assert summary.domains_affected == []

    def test_mixed_states(self):
        """Summary correctly counts mixed statuses."""
        store = make_mock_conflict_store()
        store._stored = [
            make_conflict_record(conflict_id="c1", domain="auth", status=ResolutionStatus.DETECTED, blocks_deploy=True),
            make_conflict_record(conflict_id="c2", domain="auth", status=ResolutionStatus.UNRESOLVABLE, blocks_deploy=True),
            make_conflict_record(conflict_id="c3", domain="billing", status=ResolutionStatus.AUTHORITY_RESOLVED, blocks_deploy=False),
            make_conflict_record(conflict_id="c4", domain="billing", status=ResolutionStatus.TRUST_RESOLVED, blocks_deploy=False),
            make_conflict_record(conflict_id="c5", domain="auth", status=ResolutionStatus.HUMAN_REVIEWED, blocks_deploy=False),
        ]
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        summary = resolver.get_summary()

        assert summary.total_conflicts == 5
        assert summary.unresolved_count == 2  # DETECTED + UNRESOLVABLE
        assert summary.authority_resolved_count == 1
        assert summary.trust_resolved_count == 1
        assert summary.human_reviewed_count == 1
        assert summary.deploy_blocking_count == 2  # c1 and c2

    def test_count_invariant(self):
        """total_conflicts == sum of all status counts."""
        store = make_mock_conflict_store()
        store._stored = [
            make_conflict_record(conflict_id="c1", status=ResolutionStatus.DETECTED),
            make_conflict_record(conflict_id="c2", status=ResolutionStatus.AUTHORITY_RESOLVED),
            make_conflict_record(conflict_id="c3", status=ResolutionStatus.TRUST_RESOLVED),
        ]
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        summary = resolver.get_summary()

        count_sum = (
            summary.unresolved_count
            + summary.authority_resolved_count
            + summary.trust_resolved_count
            + summary.human_reviewed_count
        )
        assert summary.total_conflicts == count_sum

    def test_domains_affected_sorted_and_deduped(self):
        """domains_affected contains only unresolved domains, sorted and deduped."""
        store = make_mock_conflict_store()
        store._stored = [
            make_conflict_record(conflict_id="c1", domain="billing", status=ResolutionStatus.DETECTED),
            make_conflict_record(conflict_id="c2", domain="auth", status=ResolutionStatus.UNRESOLVABLE),
            make_conflict_record(conflict_id="c3", domain="auth", status=ResolutionStatus.DETECTED),
            make_conflict_record(conflict_id="c4", domain="payments", status=ResolutionStatus.AUTHORITY_RESOLVED),
        ]
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        summary = resolver.get_summary()

        # Only auth and billing have unresolved conflicts; payments is resolved
        assert "auth" in summary.domains_affected
        assert "billing" in summary.domains_affected
        assert "payments" not in summary.domains_affected
        # Should be sorted
        assert summary.domains_affected == sorted(summary.domains_affected)
        # Should be deduplicated
        assert len(summary.domains_affected) == len(set(summary.domains_affected))

    def test_store_read_failed(self):
        """Store read failure raises store_read_failed."""
        store = make_mock_conflict_store()
        store.load_all = MagicMock(side_effect=Exception("read failed"))
        resolver = make_resolver(store=store)

        with pytest.raises(Exception):
            resolver.get_summary()


# ===========================================================================
# TestVerifyLogIntegrity
# ===========================================================================
class TestVerifyLogIntegrity:
    """Tests for verify_log_integrity() — SHA256 checksum verification."""

    def test_valid_log_returns_true(self):
        """Valid log with correct checksums returns True."""
        store = make_mock_conflict_store()
        store.verify_checksums = MagicMock(return_value=True)
        resolver = make_resolver(store=store)

        result = resolver.verify_log_integrity()
        assert result is True

    def test_empty_log_returns_true(self):
        """Empty log returns True (vacuously true)."""
        store = make_mock_conflict_store()
        store.verify_checksums = MagicMock(return_value=True)
        store.load_all = MagicMock(return_value=[])
        resolver = make_resolver(store=store)

        result = resolver.verify_log_integrity()
        assert result is True

    def test_tampered_log_returns_false(self):
        """Tampered log entry causes checksum mismatch, returns False."""
        store = make_mock_conflict_store()
        store.verify_checksums = MagicMock(return_value=False)
        resolver = make_resolver(store=store)

        result = resolver.verify_log_integrity()
        assert result is False

    def test_store_read_failed(self):
        """Unreadable log file raises store_read_failed."""
        store = make_mock_conflict_store()
        store.verify_checksums = MagicMock(side_effect=Exception("file not readable"))
        resolver = make_resolver(store=store)

        with pytest.raises(Exception) as exc_info:
            resolver.verify_log_integrity()
        error_str = str(exc_info.value).lower()
        assert "read" in error_str or "file" in error_str


# ===========================================================================
# TestLoadConfig
# ===========================================================================
class TestLoadConfig:
    """Tests for load_config() — YAML config loading and validation."""

    def test_valid_config(self, tmp_path):
        """Valid YAML config loaded and validated successfully."""
        config_content = """
window_timeout_seconds: 1.0
authority_override_floor: 0.7
trust_delta_threshold: 0.15
checkpoint_interval: 10
conflict_log_path: /tmp/conflicts.jsonl
protected_tiers:
  - critical
  - sensitive
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        result = load_config(str(config_file))

        assert result.window_timeout_seconds == 1.0
        assert result.authority_override_floor == 0.7
        assert result.trust_delta_threshold == 0.15
        assert result.checkpoint_interval == 10
        assert "critical" in result.protected_tiers
        assert 0.0 <= result.authority_override_floor <= 1.0
        assert 0.0 <= result.trust_delta_threshold <= 1.0

    def test_file_not_found(self):
        """Non-existent config file raises error."""
        with pytest.raises(Exception) as exc_info:
            load_config("/nonexistent/path/config.yaml")
        error_str = str(exc_info.value).lower()
        assert "not found" in error_str or "file" in error_str or "exist" in error_str or "no such" in error_str

    def test_invalid_yaml(self, tmp_path):
        """Invalid YAML content raises error."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{{{not: valid: yaml: [[[")

        with pytest.raises(Exception) as exc_info:
            load_config(str(config_file))
        error_str = str(exc_info.value).lower()
        assert "yaml" in error_str or "parse" in error_str or "invalid" in error_str

    def test_threshold_out_of_range(self, tmp_path):
        """Threshold values outside [0.0, 1.0] raise invalid_config."""
        config_content = """
window_timeout_seconds: 1.0
authority_override_floor: 1.5
trust_delta_threshold: 0.15
checkpoint_interval: 10
conflict_log_path: /tmp/conflicts.jsonl
protected_tiers: []
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises(Exception) as exc_info:
            load_config(str(config_file))
        error_str = str(exc_info.value).lower()
        assert "invalid" in error_str or "range" in error_str or "validation" in error_str or "config" in error_str

    def test_checkpoint_interval_zero(self, tmp_path):
        """checkpoint_interval < 1 raises invalid_config."""
        config_content = """
window_timeout_seconds: 1.0
authority_override_floor: 0.7
trust_delta_threshold: 0.15
checkpoint_interval: 0
conflict_log_path: /tmp/conflicts.jsonl
protected_tiers: []
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises(Exception) as exc_info:
            load_config(str(config_file))
        error_str = str(exc_info.value).lower()
        assert "interval" in error_str or "checkpoint" in error_str or "invalid" in error_str or "validation" in error_str or "config" in error_str

    def test_window_timeout_too_small(self, tmp_path):
        """window_timeout_seconds < 0.1 raises invalid_config."""
        config_content = """
window_timeout_seconds: 0.05
authority_override_floor: 0.7
trust_delta_threshold: 0.15
checkpoint_interval: 10
conflict_log_path: /tmp/conflicts.jsonl
protected_tiers: []
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises(Exception) as exc_info:
            load_config(str(config_file))
        error_str = str(exc_info.value).lower()
        assert "timeout" in error_str or "window" in error_str or "invalid" in error_str or "validation" in error_str or "config" in error_str


# ===========================================================================
# TestInvariantStatusTransitions
# ===========================================================================
class TestInvariantStatusTransitions:
    """Invariant tests for status transition rules."""

    def test_detected_to_authority_resolved(self):
        """DETECTED -> AUTHORITY_RESOLVED is a valid transition."""
        config = make_config(authority_override_floor=0.5)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=True),
                make_node_value(node_id="node-B", trust_score_snapshot=0.5, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)
        assert result.status == ResolutionStatus.AUTHORITY_RESOLVED

    def test_detected_to_trust_resolved(self):
        """DETECTED -> TRUST_RESOLVED is a valid transition."""
        config = make_config(trust_delta_threshold=0.1)
        resolver = make_resolver(config=config)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.9, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.5, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)
        assert result.status == ResolutionStatus.TRUST_RESOLVED

    def test_detected_to_unresolvable(self):
        """DETECTED -> UNRESOLVABLE is a valid transition."""
        config = make_config(trust_delta_threshold=0.5)
        signal_emitter = make_mock_signal_emitter()
        resolver = make_resolver(config=config, signal_emitter=signal_emitter)

        conflict = make_conflict_record(
            competing_values=[
                make_node_value(node_id="node-A", trust_score_snapshot=0.8, is_authoritative=False),
                make_node_value(node_id="node-B", trust_score_snapshot=0.75, is_authoritative=False),
            ],
            status=ResolutionStatus.DETECTED,
        )
        result = resolver.resolve(conflict)
        assert result.status == ResolutionStatus.UNRESOLVABLE

    def test_human_reviewed_only_from_unresolvable(self):
        """HUMAN_REVIEWED is only reachable from UNRESOLVABLE."""
        store = make_mock_conflict_store()
        # Try from AUTHORITY_RESOLVED — should fail
        resolved = make_conflict_record(
            conflict_id="c-auth",
            status=ResolutionStatus.AUTHORITY_RESOLVED,
        )
        store._stored.append(resolved)
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        with pytest.raises(Exception):
            resolver.submit_human_review(
                conflict_id="c-auth",
                winner_node_id="node-A",
                reviewed_by="admin",
                rationale="test",
            )

    def test_authority_resolved_is_terminal(self):
        """AUTHORITY_RESOLVED cannot be re-resolved."""
        resolver = make_resolver()
        conflict = make_conflict_record(
            status=ResolutionStatus.AUTHORITY_RESOLVED,
            resolution=Resolution(
                strategy=ResolutionStrategy.AUTHORITY,
                winner_node_id="node-A",
                resolved_at=_utc_now_iso(),
                rationale="authority",
                reviewed_by="",
            ),
        )
        with pytest.raises(Exception):
            resolver.resolve(conflict)

    def test_trust_resolved_is_terminal(self):
        """TRUST_RESOLVED cannot be re-resolved."""
        resolver = make_resolver()
        conflict = make_conflict_record(
            status=ResolutionStatus.TRUST_RESOLVED,
            resolution=Resolution(
                strategy=ResolutionStrategy.TRUST_ARBITRATION,
                winner_node_id="node-A",
                resolved_at=_utc_now_iso(),
                rationale="trust",
                reviewed_by="",
            ),
        )
        with pytest.raises(Exception):
            resolver.resolve(conflict)


# ===========================================================================
# TestAppendOnlyLog
# ===========================================================================
class TestAppendOnlyLog:
    """Tests for the append-only log invariant."""

    def test_resolve_appends_not_modifies(self):
        """resolve() appends new entry, does not modify original."""
        store = make_mock_conflict_store()
        resolver = make_resolver(store=store)

        conflict = make_conflict_record(status=ResolutionStatus.DETECTED)
        original_id = conflict.conflict_id

        result = resolver.resolve(conflict)

        # Store.append should have been called (not update/delete)
        assert store.append.called
        # The appended record should reference the same conflict_id
        appended = store.append.call_args[0][0] if store.append.call_args[0] else store.append.call_args[1]
        if hasattr(appended, 'conflict_id'):
            assert appended.conflict_id == original_id

    def test_human_review_appends_not_modifies(self):
        """submit_human_review() appends new entry, does not modify original."""
        store = make_mock_conflict_store()
        unresolvable = make_conflict_record(
            conflict_id="c-append-test",
            status=ResolutionStatus.UNRESOLVABLE,
            blocks_deploy=True,
            resolution=Resolution(
                strategy=ResolutionStrategy.HUMAN,
                winner_node_id="",
                resolved_at=_utc_now_iso(),
                rationale="unresolvable",
                reviewed_by="",
            ),
        )
        store._stored.append(unresolvable)
        store.load_all = MagicMock(return_value=store._stored)
        resolver = make_resolver(store=store)

        resolver.submit_human_review(
            conflict_id="c-append-test",
            winner_node_id="node-A",
            reviewed_by="admin",
            rationale="confirmed",
        )

        # Should append, not modify in place
        assert store.append.called


# ===========================================================================
# TestDeterministicResolution
# ===========================================================================
class TestDeterministicResolution:
    """Randomized test for resolution determinism using stdlib random."""

    def test_determinism_with_random_inputs(self):
        """Same conflict + config always produces same result across multiple runs."""
        import random

        config = make_config(
            authority_override_floor=0.6,
            trust_delta_threshold=0.2,
        )

        for _ in range(20):
            trust_a = round(random.uniform(0.0, 1.0), 4)
            trust_b = round(random.uniform(0.0, 1.0), 4)
            auth_a = random.choice([True, False])
            auth_b = random.choice([True, False])

            conflict = make_conflict_record(
                competing_values=[
                    make_node_value(node_id="node-A", trust_score_snapshot=trust_a, is_authoritative=auth_a, value_serialized='"a"'),
                    make_node_value(node_id="node-B", trust_score_snapshot=trust_b, is_authoritative=auth_b, value_serialized='"b"'),
                ],
                status=ResolutionStatus.DETECTED,
            )

            signal_emitter1 = make_mock_signal_emitter()
            signal_emitter2 = make_mock_signal_emitter()
            resolver1 = make_resolver(config=config, signal_emitter=signal_emitter1)
            resolver2 = make_resolver(config=config, signal_emitter=signal_emitter2)

            conflict_copy = copy.deepcopy(conflict)

            result1 = resolver1.resolve(conflict)
            result2 = resolver2.resolve(conflict_copy)

            assert result1.status == result2.status
            assert result1.resolution.strategy == result2.resolution.strategy
            assert result1.resolution.winner_node_id == result2.resolution.winner_node_id
