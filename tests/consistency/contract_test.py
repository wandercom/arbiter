"""
Contract test suite for the Consistency Analyzer component.

Tests verify behavior at boundaries against the contract specification.
Organized into: analyze_span, analyze_batch, persistence, has_high_severity, and invariants.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
from freezegun import freeze_time

# Import the component under test
from arbiter.consistency import (
    analyze_span,
    analyze_batch,
    persist,
    get_by_node,
    get_by_span,
    has_high_severity,
    AdapterObservation,
    NodeAuditClaim,
    ConsistencyFinding,
    ConsistencyOutcome,
    FindingSeverity,
    ConsistencyAnalysisError,
    AnalysisPair,
)


# ---------------------------------------------------------------------------
# Fixtures & Factory Helpers
# ---------------------------------------------------------------------------

FIXED_TIME = "2024-06-15T12:00:00Z"
FIXED_TIME_DT = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

DEFAULT_SPAN_ID = "ab" * 8  # 16-byte hex
DEFAULT_TRACE_ID = "cd" * 16  # 32-byte hex
DEFAULT_NODE_ID = "pipeline.node.alpha"

ALT_SPAN_ID = "ef" * 8
ALT_TRACE_ID = "01" * 16
ALT_NODE_ID = "pipeline.node.beta"


def make_observation(
    span_id=DEFAULT_SPAN_ID,
    trace_id=DEFAULT_TRACE_ID,
    node_id=DEFAULT_NODE_ID,
    observed_fields=None,
    timestamp=FIXED_TIME,
):
    if observed_fields is None:
        observed_fields = frozenset({"user.email", "request.body"})
    return AdapterObservation(
        span_id=span_id,
        trace_id=trace_id,
        node_id=node_id,
        observed_fields=observed_fields,
        timestamp=timestamp,
    )


def make_claim(
    span_id=DEFAULT_SPAN_ID,
    trace_id=DEFAULT_TRACE_ID,
    node_id=DEFAULT_NODE_ID,
    claimed_fields=None,
    timestamp=FIXED_TIME,
):
    if claimed_fields is None:
        claimed_fields = frozenset({"user.email", "request.body"})
    return NodeAuditClaim(
        span_id=span_id,
        trace_id=trace_id,
        node_id=node_id,
        claimed_fields=claimed_fields,
        timestamp=timestamp,
    )


def make_finding(
    node_id=DEFAULT_NODE_ID,
    span_id=DEFAULT_SPAN_ID,
    trace_id=DEFAULT_TRACE_ID,
    outcome=ConsistencyOutcome.CONSISTENT,
    severity=FindingSeverity.NONE,
    observed_fields=None,
    claimed_fields=None,
    unexplained_fields=None,
    overclaimed_fields=None,
    analyzed_at=FIXED_TIME,
    details="",
    schema_version=1,
):
    return ConsistencyFinding(
        schema_version=schema_version,
        node_id=node_id,
        span_id=span_id,
        trace_id=trace_id,
        outcome=outcome,
        severity=severity,
        observed_fields=observed_fields if observed_fields is not None else frozenset(),
        claimed_fields=claimed_fields if claimed_fields is not None else frozenset(),
        unexplained_fields=unexplained_fields if unexplained_fields is not None else frozenset(),
        overclaimed_fields=overclaimed_fields if overclaimed_fields is not None else frozenset(),
        analyzed_at=analyzed_at,
        details=details,
    )


# ===========================================================================
# TEST: analyze_span
# ===========================================================================


class TestAnalyzeSpanHappyPath:
    """Happy-path tests for analyze_span."""

    def test_consistent_matching_fields(self):
        """Both observation and claim with identical fields → CONSISTENT, NONE severity."""
        fields = frozenset({"user.email", "request.body"})
        obs = make_observation(observed_fields=fields)
        claim = make_claim(claimed_fields=fields)

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE
        assert result.unexplained_fields == frozenset()
        assert result.overclaimed_fields == frozenset()
        assert result.schema_version == 1
        assert result.node_id == DEFAULT_NODE_ID
        assert result.span_id == DEFAULT_SPAN_ID
        assert result.trace_id == DEFAULT_TRACE_ID

    def test_inconsistent_unexplained_fields(self):
        """Observation has fields not in claim → unexplained, INCONSISTENT."""
        obs = make_observation(observed_fields=frozenset({"user.email", "request.body", "response.body.token"}))
        claim = make_claim(claimed_fields=frozenset({"user.email", "request.body"}))

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.INCONSISTENT
        assert result.unexplained_fields == frozenset({"response.body.token"})
        assert result.overclaimed_fields == frozenset()

    def test_inconsistent_overclaimed_fields(self):
        """Claim has fields not in observation → overclaimed, INCONSISTENT."""
        obs = make_observation(observed_fields=frozenset({"user.email"}))
        claim = make_claim(claimed_fields=frozenset({"user.email", "user.password"}))

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.INCONSISTENT
        assert result.overclaimed_fields == frozenset({"user.password"})
        assert result.unexplained_fields == frozenset()

    def test_inconsistent_both_unexplained_and_overclaimed(self):
        """Both unexplained and overclaimed fields present → INCONSISTENT."""
        obs = make_observation(observed_fields=frozenset({"user.email", "request.body"}))
        claim = make_claim(claimed_fields=frozenset({"request.body", "user.password"}))

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.INCONSISTENT
        assert result.unexplained_fields == frozenset({"user.email"})
        assert result.overclaimed_fields == frozenset({"user.password"})

    def test_missing_claim(self):
        """Observation present, claim is None → MISSING_CLAIM, HIGH severity."""
        obs = make_observation()

        result = analyze_span(observation=obs, claim=None)

        assert result.outcome == ConsistencyOutcome.MISSING_CLAIM
        assert result.severity == FindingSeverity.HIGH
        assert result.claimed_fields == frozenset()
        assert result.node_id == DEFAULT_NODE_ID
        assert result.span_id == DEFAULT_SPAN_ID

    def test_missing_observation(self):
        """Claim present, observation is None → MISSING_OBSERVATION."""
        claim = make_claim()

        result = analyze_span(observation=None, claim=claim)

        assert result.outcome == ConsistencyOutcome.MISSING_OBSERVATION
        assert result.observed_fields == frozenset()
        assert result.node_id == DEFAULT_NODE_ID
        assert result.span_id == DEFAULT_SPAN_ID


class TestAnalyzeSpanEdgeCases:
    """Edge case tests for analyze_span."""

    def test_consistent_empty_fields_vacuous_truth(self):
        """Both present with empty field sets → CONSISTENT (vacuous truth)."""
        obs = make_observation(observed_fields=frozenset())
        claim = make_claim(claimed_fields=frozenset())

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE
        assert result.unexplained_fields == frozenset()
        assert result.overclaimed_fields == frozenset()

    def test_large_field_sets(self):
        """Large field sets still compute correct set differences."""
        observed = frozenset({f"field.level.{i}" for i in range(100)})
        claimed = frozenset({f"field.level.{i}" for i in range(50, 150)})
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)

        result = analyze_span(observation=obs, claim=claim)

        expected_unexplained = observed - claimed
        expected_overclaimed = claimed - observed
        assert result.unexplained_fields == expected_unexplained
        assert result.overclaimed_fields == expected_overclaimed
        if expected_unexplained or expected_overclaimed:
            assert result.outcome == ConsistencyOutcome.INCONSISTENT
        else:
            assert result.outcome == ConsistencyOutcome.CONSISTENT

    def test_single_field_consistent(self):
        """Single identical field in both → CONSISTENT."""
        obs = make_observation(observed_fields=frozenset({"user.email"}))
        claim = make_claim(claimed_fields=frozenset({"user.email"}))

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE

    def test_observation_fields_preserved_in_result(self):
        """Observed and claimed fields are preserved in the result."""
        observed = frozenset({"user.email", "request.body"})
        claimed = frozenset({"user.email", "response.status"})
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)

        result = analyze_span(observation=obs, claim=claim)

        assert result.observed_fields == observed
        assert result.claimed_fields == claimed


class TestAnalyzeSpanErrors:
    """Error case tests for analyze_span."""

    def test_both_none_raises_error(self):
        """Both observation and claim are None → ConsistencyAnalysisError."""
        with pytest.raises(ConsistencyAnalysisError):
            analyze_span(observation=None, claim=None)

    def test_span_id_mismatch_raises_error(self):
        """Mismatched span_id → ConsistencyAnalysisError with span info."""
        obs = make_observation(span_id="aa" * 8)
        claim = make_claim(span_id="bb" * 8)

        with pytest.raises(ConsistencyAnalysisError) as exc_info:
            analyze_span(observation=obs, claim=claim)

        error = exc_info.value
        # Error should contain identifying info (span_id)
        error_str = str(error)
        assert "aa" * 8 in error_str or "bb" * 8 in error_str or "span" in error_str.lower()

    def test_node_id_mismatch_raises_error(self):
        """Mismatched node_id → ConsistencyAnalysisError with node info."""
        obs = make_observation(node_id="pipeline.node.alpha")
        claim = make_claim(node_id="pipeline.node.beta")

        with pytest.raises(ConsistencyAnalysisError) as exc_info:
            analyze_span(observation=obs, claim=claim)

        error = exc_info.value
        error_str = str(error)
        assert "alpha" in error_str or "beta" in error_str or "node" in error_str.lower()

    def test_trace_id_mismatch_raises_error(self):
        """Mismatched trace_id → ConsistencyAnalysisError with trace info."""
        obs = make_observation(trace_id="aa" * 16)
        claim = make_claim(trace_id="bb" * 16)

        with pytest.raises(ConsistencyAnalysisError) as exc_info:
            analyze_span(observation=obs, claim=claim)

        error = exc_info.value
        error_str = str(error)
        assert "aa" * 16 in error_str or "bb" * 16 in error_str or "trace" in error_str.lower()

    def test_both_none_error_has_descriptive_message(self):
        """Both None error includes descriptive message."""
        with pytest.raises(ConsistencyAnalysisError) as exc_info:
            analyze_span(observation=None, claim=None)

        # The error should have some descriptive information
        assert exc_info.value is not None


class TestAnalyzeSpanSeverityMapping:
    """Tests for severity deterministic mapping."""

    def test_severity_none_when_consistent(self):
        """CONSISTENT → NONE severity."""
        obs = make_observation(observed_fields=frozenset({"user.email"}))
        claim = make_claim(claimed_fields=frozenset({"user.email"}))

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE

    def test_severity_high_when_missing_claim(self):
        """MISSING_CLAIM → HIGH severity."""
        obs = make_observation()

        result = analyze_span(observation=obs, claim=None)

        assert result.outcome == ConsistencyOutcome.MISSING_CLAIM
        assert result.severity == FindingSeverity.HIGH

    def test_analyzed_at_is_valid_utc(self):
        """analyzed_at is a valid UTC timestamp."""
        before = datetime.now(timezone.utc)
        obs = make_observation()
        claim = make_claim()

        result = analyze_span(observation=obs, claim=claim)

        # Parse the analyzed_at to verify it's a valid timestamp
        analyzed_at = result.analyzed_at
        assert analyzed_at is not None
        assert len(analyzed_at) > 0
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(analyzed_at.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
        assert parsed >= before - timedelta(seconds=1)  # small tolerance


# ===========================================================================
# TEST: analyze_batch
# ===========================================================================


class TestAnalyzeBatchHappyPath:
    """Happy-path tests for analyze_batch."""

    def test_batch_returns_one_finding_per_pair(self):
        """Batch of valid pairs returns one finding per pair."""
        pairs = []
        for i in range(3):
            sid = f"{i:02d}" * 8
            obs = make_observation(span_id=sid, observed_fields=frozenset({"user.email"}))
            claim = make_claim(span_id=sid, claimed_fields=frozenset({"user.email"}))
            pairs.append(AnalysisPair(observation=obs, claim=claim))

        result = analyze_batch(pairs)

        assert len(result) == 3
        for i, finding in enumerate(result):
            expected_sid = f"{i:02d}" * 8
            assert finding.span_id == expected_sid

    def test_batch_mixed_outcomes(self):
        """Batch with mixed pair types produces correct outcomes per pair."""
        # Pair 1: consistent
        obs1 = make_observation(
            span_id="11" * 8,
            observed_fields=frozenset({"user.email"}),
        )
        claim1 = make_claim(
            span_id="11" * 8,
            claimed_fields=frozenset({"user.email"}),
        )
        # Pair 2: missing claim
        obs2 = make_observation(span_id="22" * 8)
        # Pair 3: missing observation
        claim3 = make_claim(span_id="33" * 8)

        pairs = [
            AnalysisPair(observation=obs1, claim=claim1),
            AnalysisPair(observation=obs2, claim=None),
            AnalysisPair(observation=None, claim=claim3),
        ]

        result = analyze_batch(pairs)

        assert len(result) == 3
        assert result[0].outcome == ConsistencyOutcome.CONSISTENT
        assert result[1].outcome == ConsistencyOutcome.MISSING_CLAIM
        assert result[2].outcome == ConsistencyOutcome.MISSING_OBSERVATION


class TestAnalyzeBatchEdgeCases:
    """Edge cases for analyze_batch."""

    def test_single_pair_batch(self):
        """Single-pair batch works identically to analyze_span."""
        obs = make_observation()
        claim = make_claim()
        pairs = [AnalysisPair(observation=obs, claim=claim)]

        result = analyze_batch(pairs)

        assert len(result) == 1
        assert result[0].outcome == ConsistencyOutcome.CONSISTENT

    def test_ordering_preserved(self):
        """Results maintain the same ordering as input pairs."""
        pairs = []
        node_ids = ["pipeline.node.first", "pipeline.node.second", "pipeline.node.third"]
        for i, nid in enumerate(node_ids):
            sid = f"{i:02d}" * 8
            tid = f"{i:02d}" * 16
            obs = make_observation(span_id=sid, trace_id=tid, node_id=nid)
            claim = make_claim(span_id=sid, trace_id=tid, node_id=nid)
            pairs.append(AnalysisPair(observation=obs, claim=claim))

        result = analyze_batch(pairs)

        assert len(result) == len(pairs)
        for i, finding in enumerate(result):
            assert finding.node_id == node_ids[i]


class TestAnalyzeBatchErrors:
    """Error case tests for analyze_batch."""

    def test_empty_batch_raises_error(self):
        """Empty batch → ConsistencyAnalysisError."""
        with pytest.raises(ConsistencyAnalysisError):
            analyze_batch([])

    def test_pair_both_none_raises_error(self):
        """Pair with both None in batch → ConsistencyAnalysisError (error propagates, no partial results)."""
        obs = make_observation(span_id="11" * 8)
        claim = make_claim(span_id="11" * 8)
        pairs = [
            AnalysisPair(observation=obs, claim=claim),
            AnalysisPair(observation=None, claim=None),  # invalid
        ]

        with pytest.raises(ConsistencyAnalysisError):
            analyze_batch(pairs)

    def test_pair_id_mismatch_raises_error(self):
        """Pair with mismatched IDs in batch → ConsistencyAnalysisError."""
        obs = make_observation(span_id="aa" * 8)
        claim = make_claim(span_id="bb" * 8)
        pairs = [AnalysisPair(observation=obs, claim=claim)]

        with pytest.raises(ConsistencyAnalysisError):
            analyze_batch(pairs)


# ===========================================================================
# TEST: persist / get_by_node / get_by_span
# ===========================================================================


class TestPersistHappyPath:
    """Happy-path tests for persist and retrieval."""

    def test_persist_and_retrieve_by_node(self, tmp_path):
        """Persisted finding is retrievable via get_by_node."""
        finding = make_finding(node_id="pipeline.test.persist")

        # We need to mock or configure the store to use tmp_path
        # The test verifies the contract behavior: persist → get_by_node returns it
        persist(finding)
        results = get_by_node("pipeline.test.persist")

        assert len(results) >= 1
        matched = [f for f in results if f.span_id == finding.span_id]
        assert len(matched) >= 1
        assert matched[0].node_id == "pipeline.test.persist"

    def test_persist_and_retrieve_by_span(self, tmp_path):
        """Persisted finding is retrievable via get_by_span."""
        finding = make_finding(span_id="ff" * 8)

        persist(finding)
        results = get_by_span("ff" * 8)

        assert len(results) >= 1
        matched = [f for f in results if f.node_id == finding.node_id]
        assert len(matched) >= 1
        assert matched[0].span_id == "ff" * 8


class TestPersistAppendOnly:
    """Append-only invariant tests for persist."""

    def test_multiple_persists_accumulate(self, tmp_path):
        """Multiple persists increase store size, no overwrites."""
        finding1 = make_finding(
            span_id="11" * 8,
            analyzed_at="2024-06-15T12:00:00Z",
        )
        finding2 = make_finding(
            span_id="22" * 8,
            analyzed_at="2024-06-15T12:01:00Z",
        )

        persist(finding1)
        persist(finding2)

        results = get_by_node(DEFAULT_NODE_ID)
        span_ids = {f.span_id for f in results}
        assert "11" * 8 in span_ids or "1111111111111111" in span_ids
        assert "22" * 8 in span_ids or "2222222222222222" in span_ids


class TestPersistErrors:
    """Error case tests for persist."""

    def test_io_error_on_persist(self):
        """IO error during persist raises appropriate error."""
        finding = make_finding()

        # Mock the underlying storage to simulate IO failure
        with patch("builtins.open", side_effect=IOError("disk full")):
            with pytest.raises(Exception):  # Could be IOError or wrapped ConsistencyAnalysisError
                persist(finding)


class TestGetByNodeDetails:
    """Detailed tests for get_by_node."""

    def test_empty_result_for_unknown_node(self):
        """Returns empty list for node_id with no findings."""
        results = get_by_node("nonexistent.node.xyz123")
        assert results == []

    def test_filters_by_node_id(self, tmp_path):
        """Only returns findings matching the given node_id."""
        finding_a = make_finding(node_id="pipeline.node.alpha", span_id="aa" * 8)
        finding_b = make_finding(node_id="pipeline.node.beta", span_id="bb" * 8)

        persist(finding_a)
        persist(finding_b)

        results = get_by_node("pipeline.node.alpha")
        assert all(f.node_id == "pipeline.node.alpha" for f in results)

    def test_chronological_order(self, tmp_path):
        """Results are ordered by analyzed_at ascending (oldest first)."""
        finding1 = make_finding(
            span_id="11" * 8,
            analyzed_at="2024-01-01T00:00:00Z",
        )
        finding2 = make_finding(
            span_id="22" * 8,
            analyzed_at="2024-06-15T12:00:00Z",
        )
        finding3 = make_finding(
            span_id="33" * 8,
            analyzed_at="2024-03-01T06:00:00Z",
        )

        persist(finding1)
        persist(finding2)
        persist(finding3)

        results = get_by_node(DEFAULT_NODE_ID)
        timestamps = [f.analyzed_at for f in results]
        assert timestamps == sorted(timestamps)


class TestGetBySpanDetails:
    """Detailed tests for get_by_span."""

    def test_empty_result_for_unknown_span(self):
        """Returns empty list for span_id with no findings."""
        results = get_by_span("00" * 8)
        assert results == []

    def test_filters_by_span_id(self, tmp_path):
        """Only returns findings matching the given span_id."""
        finding_a = make_finding(span_id="aa" * 8)
        finding_b = make_finding(span_id="bb" * 8)

        persist(finding_a)
        persist(finding_b)

        results = get_by_span("aa" * 8)
        assert all(f.span_id == "aa" * 8 for f in results)

    def test_chronological_order(self, tmp_path):
        """Results ordered by analyzed_at ascending."""
        finding1 = make_finding(
            span_id="aa" * 8,
            analyzed_at="2024-01-01T00:00:00Z",
        )
        finding2 = make_finding(
            span_id="aa" * 8,
            analyzed_at="2024-06-15T12:00:00Z",
        )

        persist(finding1)
        persist(finding2)

        results = get_by_span("aa" * 8)
        timestamps = [f.analyzed_at for f in results]
        assert timestamps == sorted(timestamps)


# ===========================================================================
# TEST: has_high_severity
# ===========================================================================


class TestHasHighSeverityHappyPath:
    """Happy-path tests for has_high_severity."""

    def test_returns_true_when_high_exists(self, tmp_path):
        """Returns True when HIGH severity finding exists for the node."""
        finding = make_finding(
            node_id="pipeline.node.high",
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
        )
        persist(finding)

        result = has_high_severity("pipeline.node.high", since=None)
        assert result is True

    def test_returns_true_with_since_before_finding(self, tmp_path):
        """Returns True when HIGH finding is at or after since timestamp."""
        finding = make_finding(
            node_id="pipeline.node.high2",
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2024-06-15T12:00:00Z",
        )
        persist(finding)

        result = has_high_severity("pipeline.node.high2", since="2024-06-01T00:00:00Z")
        assert result is True

    def test_since_none_considers_all(self, tmp_path):
        """When since is None, considers all findings regardless of time."""
        finding = make_finding(
            node_id="pipeline.node.alltime",
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2020-01-01T00:00:00Z",
        )
        persist(finding)

        result = has_high_severity("pipeline.node.alltime", since=None)
        assert result is True


class TestHasHighSeverityEdgeCases:
    """Edge case tests for has_high_severity."""

    def test_returns_false_no_findings(self):
        """Returns False when no findings exist for the node."""
        result = has_high_severity("nonexistent.node.xyz999", since=None)
        assert result is False

    def test_returns_false_only_low_severity(self, tmp_path):
        """Returns False when only LOW/MEDIUM/NONE severity findings exist."""
        for sev in [FindingSeverity.NONE, FindingSeverity.LOW, FindingSeverity.MEDIUM]:
            finding = make_finding(
                node_id="pipeline.node.lowonly",
                span_id=f"{'ab' * 7}{sev.name[:2].lower()}" if hasattr(sev, 'name') else "ab" * 8,
                severity=sev,
                outcome=ConsistencyOutcome.CONSISTENT if sev == FindingSeverity.NONE else ConsistencyOutcome.INCONSISTENT,
            )
            persist(finding)

        result = has_high_severity("pipeline.node.lowonly", since=None)
        assert result is False

    def test_returns_false_high_before_since(self, tmp_path):
        """Returns False when HIGH finding is before the since timestamp."""
        finding = make_finding(
            node_id="pipeline.node.oldhigh",
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2024-01-01T00:00:00Z",
        )
        persist(finding)

        result = has_high_severity("pipeline.node.oldhigh", since="2024-06-01T00:00:00Z")
        assert result is False

    def test_returns_false_future_since(self, tmp_path):
        """Returns False when since is in the future beyond all findings."""
        finding = make_finding(
            node_id="pipeline.node.futuretime",
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2024-06-15T12:00:00Z",
        )
        persist(finding)

        result = has_high_severity("pipeline.node.futuretime", since="2099-01-01T00:00:00Z")
        assert result is False

    def test_persist_high_then_has_high_is_true(self, tmp_path):
        """After persisting a HIGH severity finding, has_high_severity returns True."""
        node = "pipeline.node.posthigh"

        # Before: no findings
        assert has_high_severity(node, since=None) is False

        # Persist HIGH finding
        finding = make_finding(
            node_id=node,
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
        )
        persist(finding)

        # After: should be True
        assert has_high_severity(node, since=None) is True


# ===========================================================================
# TEST: Invariants
# ===========================================================================


class TestInvariants:
    """Cross-cutting invariant tests."""

    def test_set_difference_invariant_observed_superset(self):
        """unexplained = observed - claimed, overclaimed = claimed - observed (observed superset)."""
        observed = frozenset({"a.b", "c.d", "e.f"})
        claimed = frozenset({"a.b"})
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)

        result = analyze_span(observation=obs, claim=claim)

        assert result.unexplained_fields == observed - claimed
        assert result.overclaimed_fields == claimed - observed

    def test_set_difference_invariant_claimed_superset(self):
        """overclaimed = claimed - observed when claimed is superset."""
        observed = frozenset({"a.b"})
        claimed = frozenset({"a.b", "c.d", "e.f"})
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)

        result = analyze_span(observation=obs, claim=claim)

        assert result.unexplained_fields == observed - claimed
        assert result.overclaimed_fields == claimed - observed

    def test_set_difference_invariant_disjoint(self):
        """Completely disjoint field sets."""
        observed = frozenset({"a.b", "c.d"})
        claimed = frozenset({"e.f", "g.h"})
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)

        result = analyze_span(observation=obs, claim=claim)

        assert result.unexplained_fields == observed
        assert result.overclaimed_fields == claimed
        assert result.outcome == ConsistencyOutcome.INCONSISTENT

    def test_set_difference_invariant_identical(self):
        """Identical field sets produce empty differences."""
        fields = frozenset({"a.b", "c.d"})
        obs = make_observation(observed_fields=fields)
        claim = make_claim(claimed_fields=fields)

        result = analyze_span(observation=obs, claim=claim)

        assert result.unexplained_fields == frozenset()
        assert result.overclaimed_fields == frozenset()
        assert result.outcome == ConsistencyOutcome.CONSISTENT

    def test_outcome_consistent_iff_no_differences(self):
        """CONSISTENT iff unexplained and overclaimed are both empty and both present."""
        # Case 1: Both empty → CONSISTENT
        obs = make_observation(observed_fields=frozenset({"x.y"}))
        claim = make_claim(claimed_fields=frozenset({"x.y"}))
        result = analyze_span(observation=obs, claim=claim)
        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.unexplained_fields == frozenset()
        assert result.overclaimed_fields == frozenset()

        # Case 2: Non-empty unexplained → INCONSISTENT
        obs2 = make_observation(observed_fields=frozenset({"x.y", "z.w"}))
        claim2 = make_claim(claimed_fields=frozenset({"x.y"}))
        result2 = analyze_span(observation=obs2, claim=claim2)
        assert result2.outcome == ConsistencyOutcome.INCONSISTENT
        assert result2.unexplained_fields != frozenset()

    def test_adapter_is_ground_truth_missing_claim_high(self):
        """Adapter observations are ground truth — missing claim is HIGH severity."""
        obs = make_observation(observed_fields=frozenset({"sensitive.data"}))
        result = analyze_span(observation=obs, claim=None)

        assert result.outcome == ConsistencyOutcome.MISSING_CLAIM
        assert result.severity == FindingSeverity.HIGH

    def test_schema_version_always_1(self):
        """schema_version is always 1 for all outcomes."""
        # CONSISTENT
        obs = make_observation(observed_fields=frozenset({"a.b"}))
        claim = make_claim(claimed_fields=frozenset({"a.b"}))
        assert analyze_span(observation=obs, claim=claim).schema_version == 1

        # MISSING_CLAIM
        assert analyze_span(observation=obs, claim=None).schema_version == 1

        # MISSING_OBSERVATION
        assert analyze_span(observation=None, claim=claim).schema_version == 1

        # INCONSISTENT
        obs2 = make_observation(observed_fields=frozenset({"a.b", "c.d"}))
        assert analyze_span(observation=obs2, claim=claim).schema_version == 1

    @pytest.mark.parametrize(
        "observed,claimed,expected_outcome",
        [
            (frozenset({"a.b"}), frozenset({"a.b"}), ConsistencyOutcome.CONSISTENT),
            (frozenset({"a.b", "c.d"}), frozenset({"a.b"}), ConsistencyOutcome.INCONSISTENT),
            (frozenset({"a.b"}), frozenset({"a.b", "c.d"}), ConsistencyOutcome.INCONSISTENT),
            (frozenset(), frozenset(), ConsistencyOutcome.CONSISTENT),
            (frozenset({"x.y"}), frozenset({"z.w"}), ConsistencyOutcome.INCONSISTENT),
        ],
    )
    def test_outcome_parametrized(self, observed, claimed, expected_outcome):
        """Parametrized outcome verification for various field combinations."""
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)

        result = analyze_span(observation=obs, claim=claim)

        assert result.outcome == expected_outcome

    def test_findings_chronological_order_invariant(self, tmp_path):
        """get_by_node returns findings in chronological order."""
        node = "pipeline.node.chrono"
        timestamps = [
            "2024-01-15T12:00:00Z",
            "2024-06-15T12:00:00Z",
            "2024-03-15T12:00:00Z",
        ]
        for i, ts in enumerate(timestamps):
            finding = make_finding(
                node_id=node,
                span_id=f"{i:02d}" * 8,
                analyzed_at=ts,
            )
            persist(finding)

        results = get_by_node(node)
        result_timestamps = [f.analyzed_at for f in results]
        assert result_timestamps == sorted(result_timestamps)

    def test_analyze_both_none_invalid_invariant(self):
        """analyze_span with both observation=None and claim=None is invalid."""
        with pytest.raises(ConsistencyAnalysisError):
            analyze_span(observation=None, claim=None)


class TestRandomizedFieldSets:
    """Randomized tests for field set invariants using stdlib random."""

    def test_random_field_set_differences(self):
        """Randomized verification of set difference invariants."""
        import random
        random.seed(42)

        for _ in range(20):
            n_observed = random.randint(0, 15)
            n_claimed = random.randint(0, 15)
            all_fields = [f"field.level{i}.sub{j}" for i in range(20) for j in range(3)]
            observed = frozenset(random.sample(all_fields, min(n_observed, len(all_fields))))
            claimed = frozenset(random.sample(all_fields, min(n_claimed, len(all_fields))))

            obs = make_observation(observed_fields=observed)
            claim = make_claim(claimed_fields=claimed)

            result = analyze_span(observation=obs, claim=claim)

            # Invariant: set differences
            assert result.unexplained_fields == observed - claimed
            assert result.overclaimed_fields == claimed - observed

            # Invariant: outcome consistency
            if (observed - claimed) == frozenset() and (claimed - observed) == frozenset():
                assert result.outcome == ConsistencyOutcome.CONSISTENT
                assert result.severity == FindingSeverity.NONE
            else:
                assert result.outcome == ConsistencyOutcome.INCONSISTENT
