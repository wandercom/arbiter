"""
Adversarial hidden acceptance tests for the Consistency Analyzer component.

These tests target behavioral gaps not covered by visible tests, designed to
catch implementations that hardcode returns or take shortcuts based on visible
test inputs.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import time

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
    ConsistencyAnalysisError,
    ConsistencyOutcome,
    FindingSeverity,
    AnalysisPair,
)


# ---- Helpers ----

def make_observation(
    span_id="aa" * 8,
    trace_id="bb" * 16,
    node_id="test.node",
    observed_fields=frozenset(),
    timestamp="2024-01-15T12:00:00Z",
):
    return AdapterObservation(
        span_id=span_id,
        trace_id=trace_id,
        node_id=node_id,
        observed_fields=observed_fields,
        timestamp=timestamp,
    )


def make_claim(
    span_id="aa" * 8,
    trace_id="bb" * 16,
    node_id="test.node",
    claimed_fields=frozenset(),
    timestamp="2024-01-15T12:00:00Z",
):
    return NodeAuditClaim(
        span_id=span_id,
        trace_id=trace_id,
        node_id=node_id,
        claimed_fields=claimed_fields,
        timestamp=timestamp,
    )


def make_finding(
    node_id="test.node",
    span_id="aa" * 8,
    trace_id="bb" * 16,
    outcome=ConsistencyOutcome.CONSISTENT,
    severity=FindingSeverity.NONE,
    observed_fields=frozenset(),
    claimed_fields=frozenset(),
    unexplained_fields=frozenset(),
    overclaimed_fields=frozenset(),
    analyzed_at="2024-01-15T12:00:00Z",
    details="",
):
    return ConsistencyFinding(
        schema_version=1,
        node_id=node_id,
        span_id=span_id,
        trace_id=trace_id,
        outcome=outcome,
        severity=severity,
        observed_fields=observed_fields,
        claimed_fields=claimed_fields,
        unexplained_fields=unexplained_fields,
        overclaimed_fields=overclaimed_fields,
        analyzed_at=analyzed_at,
        details=details,
    )


# ---- Tests ----


class TestGoodhartAnalyzeSpan:
    """Adversarial tests for analyze_span."""

    def test_goodhart_node_id_span_id_trace_id_propagated(self):
        """The finding's node_id, span_id, and trace_id must be copied from the observation (or claim) inputs, not hardcoded to any particular test value"""
        obs = make_observation(
            node_id="alpha.bravo.charlie",
            span_id="ab" * 8,
            trace_id="cd" * 16,
            observed_fields=frozenset({"x.y"}),
        )
        claim = make_claim(
            node_id="alpha.bravo.charlie",
            span_id="ab" * 8,
            trace_id="cd" * 16,
            claimed_fields=frozenset({"x.y"}),
        )
        result = analyze_span(observation=obs, claim=claim)
        assert result.node_id == "alpha.bravo.charlie"
        assert result.span_id == "ab" * 8
        assert result.trace_id == "cd" * 16

    def test_goodhart_large_field_sets(self):
        """Set difference computation must work correctly for large field sets, not just small hand-picked examples"""
        observed = frozenset(f"field.{i}" for i in range(100))
        claimed = frozenset(f"field.{i}" for i in range(50, 130))
        obs = make_observation(observed_fields=observed)
        claim = make_claim(claimed_fields=claimed)
        result = analyze_span(observation=obs, claim=claim)
        assert len(result.unexplained_fields) == 50  # field.0 - field.49
        assert len(result.overclaimed_fields) == 30  # field.100 - field.129
        assert result.outcome == ConsistencyOutcome.INCONSISTENT

    def test_goodhart_subset_claim_overclaim_only(self):
        """When claimed_fields is a strict superset of observed_fields, unexplained should be empty and overclaimed should be non-empty, with outcome INCONSISTENT"""
        obs = make_observation(observed_fields=frozenset({"a.b", "c.d"}))
        claim = make_claim(claimed_fields=frozenset({"a.b", "c.d", "e.f", "g.h"}))
        result = analyze_span(observation=obs, claim=claim)
        assert len(result.unexplained_fields) == 0
        assert len(result.overclaimed_fields) == 2
        assert result.outcome == ConsistencyOutcome.INCONSISTENT

    def test_goodhart_subset_observation_unexplained_only(self):
        """When observed_fields is a strict superset of claimed_fields, overclaimed should be empty and unexplained should be non-empty"""
        obs = make_observation(
            observed_fields=frozenset({"x.a", "x.b", "x.c", "x.d", "x.e"})
        )
        claim = make_claim(claimed_fields=frozenset({"x.d", "x.e"}))
        result = analyze_span(observation=obs, claim=claim)
        assert len(result.overclaimed_fields) == 0
        assert len(result.unexplained_fields) == 3
        assert result.outcome == ConsistencyOutcome.INCONSISTENT

    def test_goodhart_missing_claim_fields_in_finding(self):
        """When observation is present but claim is None, the finding must carry the observation's fields as observed_fields, empty claimed_fields, and observed_fields as unexplained_fields"""
        obs = make_observation(
            observed_fields=frozenset({"request.body", "response.token", "user.id"})
        )
        result = analyze_span(observation=obs, claim=None)
        assert result.observed_fields == frozenset(
            {"request.body", "response.token", "user.id"}
        )
        assert result.claimed_fields == frozenset()
        assert result.unexplained_fields == frozenset(
            {"request.body", "response.token", "user.id"}
        )
        assert result.overclaimed_fields == frozenset()

    def test_goodhart_missing_observation_fields_in_finding(self):
        """When claim is present but observation is None, the finding must carry the claim's fields as claimed_fields, empty observed_fields, and claimed_fields as overclaimed_fields"""
        claim = make_claim(
            claimed_fields=frozenset({"auth.jwt", "db.query"})
        )
        result = analyze_span(observation=None, claim=claim)
        assert result.claimed_fields == frozenset({"auth.jwt", "db.query"})
        assert result.observed_fields == frozenset()
        assert result.overclaimed_fields == frozenset({"auth.jwt", "db.query"})
        assert result.unexplained_fields == frozenset()

    def test_goodhart_missing_observation_severity_not_none_not_high(self):
        """MISSING_OBSERVATION outcome severity must be deterministic and distinct from NONE (CONSISTENT) and HIGH (MISSING_CLAIM)"""
        claim = make_claim(claimed_fields=frozenset({"a.b", "c.d"}))
        result = analyze_span(observation=None, claim=claim)
        assert result.severity in (FindingSeverity.LOW, FindingSeverity.MEDIUM)
        assert result.outcome == ConsistencyOutcome.MISSING_OBSERVATION

    def test_goodhart_severity_scales_with_unexplained_count(self):
        """Severity for INCONSISTENT outcomes must scale deterministically with the number of unexplained fields"""
        # One unexplained field
        obs_one = make_observation(
            observed_fields=frozenset({"a.b"}),
            span_id="11" * 8,
        )
        claim_one = make_claim(
            claimed_fields=frozenset(),
            span_id="11" * 8,
        )
        result_one = analyze_span(observation=obs_one, claim=claim_one)

        # Ten unexplained fields
        obs_many = make_observation(
            observed_fields=frozenset(f"field.{i}" for i in range(10)),
            span_id="22" * 8,
        )
        claim_many = make_claim(
            claimed_fields=frozenset(),
            span_id="22" * 8,
        )
        result_many = analyze_span(observation=obs_many, claim=claim_many)

        assert result_one.severity in (FindingSeverity.LOW, FindingSeverity.MEDIUM)
        assert result_many.severity == FindingSeverity.HIGH

    def test_goodhart_overclaim_only_severity_low(self):
        """When the only difference is overclaimed fields (no unexplained), severity should be LOW"""
        obs = make_observation(observed_fields=frozenset({"a.b"}))
        claim = make_claim(claimed_fields=frozenset({"a.b", "c.d"}))
        result = analyze_span(observation=obs, claim=claim)
        assert result.outcome == ConsistencyOutcome.INCONSISTENT
        assert result.severity == FindingSeverity.LOW

    def test_goodhart_details_nonempty_on_inconsistent(self):
        """The details field should contain meaningful diagnostic information when the outcome is not CONSISTENT"""
        obs = make_observation(observed_fields=frozenset({"a.b"}))
        claim = make_claim(claimed_fields=frozenset({"c.d"}))
        result = analyze_span(observation=obs, claim=claim)
        assert len(result.details) > 0
        assert result.outcome == ConsistencyOutcome.INCONSISTENT

    def test_goodhart_deeply_nested_fields(self):
        """Dot-notation field names with deep nesting (3+ levels) must be treated as valid and participate correctly in set differences"""
        fields = frozenset(
            {"a.b.c.d.e.f", "response.body.data.items.0.id", "x.y.z.w"}
        )
        obs = make_observation(observed_fields=fields)
        claim = make_claim(claimed_fields=fields)
        result = analyze_span(observation=obs, claim=claim)
        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE

    def test_goodhart_single_field_consistent(self):
        """A single-field observation matching a single-field claim must be CONSISTENT"""
        obs = make_observation(observed_fields=frozenset({"only.field"}))
        claim = make_claim(claimed_fields=frozenset({"only.field"}))
        result = analyze_span(observation=obs, claim=claim)
        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE
        assert len(result.unexplained_fields) == 0
        assert len(result.overclaimed_fields) == 0

    def test_goodhart_disjoint_fields(self):
        """When observation and claim have completely disjoint field sets, all observed are unexplained and all claimed are overclaimed"""
        obs = make_observation(observed_fields=frozenset({"obs.a", "obs.b"}))
        claim = make_claim(
            claimed_fields=frozenset({"claim.x", "claim.y", "claim.z"})
        )
        result = analyze_span(observation=obs, claim=claim)
        assert result.unexplained_fields == frozenset({"obs.a", "obs.b"})
        assert result.overclaimed_fields == frozenset(
            {"claim.x", "claim.y", "claim.z"}
        )
        assert result.outcome == ConsistencyOutcome.INCONSISTENT

    def test_goodhart_error_includes_context(self):
        """Error messages from ConsistencyAnalysisError must include identifying context"""
        obs = make_observation(node_id="unique.node.xyz")
        claim = make_claim(node_id="different.node.abc")
        with pytest.raises(ConsistencyAnalysisError) as exc_info:
            analyze_span(observation=obs, claim=claim)
        error = exc_info.value
        # Error must include some context about the problematic IDs
        error_str = str(error)
        assert "unique.node.xyz" in error_str or "different.node.abc" in error_str or (
            hasattr(error, "node_id") and error.node_id != ""
        )

    def test_goodhart_missing_claim_empty_observation_fields(self):
        """MISSING_CLAIM with empty observed_fields should still produce HIGH severity"""
        obs = make_observation(observed_fields=frozenset())
        result = analyze_span(observation=obs, claim=None)
        assert result.outcome == ConsistencyOutcome.MISSING_CLAIM
        assert result.severity == FindingSeverity.HIGH
        assert len(result.observed_fields) == 0

    def test_goodhart_field_names_case_sensitive(self):
        """Field name comparison must be case-sensitive"""
        obs = make_observation(observed_fields=frozenset({"user.email"}))
        claim = make_claim(claimed_fields=frozenset({"User.Email"}))
        result = analyze_span(observation=obs, claim=claim)
        assert result.outcome == ConsistencyOutcome.INCONSISTENT
        assert len(result.unexplained_fields) == 1
        assert len(result.overclaimed_fields) == 1

    def test_goodhart_observed_claimed_fields_preserved(self):
        """The finding must preserve the original observed_fields and claimed_fields exactly"""
        obs = make_observation(
            observed_fields=frozenset({"a.b", "c.d", "e.f"})
        )
        claim = make_claim(
            claimed_fields=frozenset({"c.d", "e.f", "g.h"})
        )
        result = analyze_span(observation=obs, claim=claim)
        assert result.observed_fields == frozenset({"a.b", "c.d", "e.f"})
        assert result.claimed_fields == frozenset({"c.d", "e.f", "g.h"})

    def test_goodhart_consistent_many_fields(self):
        """CONSISTENT outcome must hold for larger identical field sets"""
        fields = frozenset(f"field.{i}" for i in range(20))
        obs = make_observation(observed_fields=fields)
        claim = make_claim(claimed_fields=fields)
        result = analyze_span(observation=obs, claim=claim)
        assert result.outcome == ConsistencyOutcome.CONSISTENT
        assert result.severity == FindingSeverity.NONE
        assert len(result.observed_fields) == 20

    def test_goodhart_trace_id_in_result(self):
        """The finding must carry trace_id from the inputs"""
        obs = make_observation(trace_id="ef" * 16, observed_fields=frozenset({"a.b"}))
        claim = make_claim(trace_id="ef" * 16, claimed_fields=frozenset({"a.b"}))
        result = analyze_span(observation=obs, claim=claim)
        assert result.trace_id == "ef" * 16

    def test_goodhart_missing_observation_severity_not_none(self):
        """MISSING_OBSERVATION should not have NONE severity"""
        claim = make_claim(claimed_fields=frozenset({"a.b", "c.d", "e.f"}))
        result = analyze_span(observation=None, claim=claim)
        assert result.severity != FindingSeverity.NONE
        assert result.outcome == ConsistencyOutcome.MISSING_OBSERVATION


class TestGoodhartAnalyzeBatch:
    """Adversarial tests for analyze_batch."""

    def test_goodhart_mixed_outcomes(self):
        """A batch containing pairs that produce different outcomes must return correctly typed findings for each"""
        pairs = [
            AnalysisPair(
                observation=make_observation(
                    span_id="01" * 8,
                    observed_fields=frozenset({"a.b"}),
                ),
                claim=make_claim(
                    span_id="01" * 8,
                    claimed_fields=frozenset({"a.b"}),
                ),
            ),
            AnalysisPair(
                observation=make_observation(
                    span_id="02" * 8,
                    observed_fields=frozenset({"a.b"}),
                ),
                claim=make_claim(
                    span_id="02" * 8,
                    claimed_fields=frozenset({"c.d"}),
                ),
            ),
            AnalysisPair(
                observation=make_observation(
                    span_id="03" * 8,
                    observed_fields=frozenset({"a.b"}),
                ),
                claim=None,
            ),
            AnalysisPair(
                observation=None,
                claim=make_claim(
                    span_id="04" * 8,
                    claimed_fields=frozenset({"a.b"}),
                ),
            ),
        ]
        results = analyze_batch(pairs)
        assert len(results) == 4
        assert results[0].outcome == ConsistencyOutcome.CONSISTENT
        assert results[1].outcome == ConsistencyOutcome.INCONSISTENT
        assert results[2].outcome == ConsistencyOutcome.MISSING_CLAIM
        assert results[3].outcome == ConsistencyOutcome.MISSING_OBSERVATION

    def test_goodhart_large_batch(self):
        """Batch processing must handle non-trivial batch sizes correctly"""
        pairs = [
            AnalysisPair(
                observation=make_observation(
                    span_id=f"{i:016x}",
                    observed_fields=frozenset({"a.b"}),
                ),
                claim=make_claim(
                    span_id=f"{i:016x}",
                    claimed_fields=frozenset({"a.b"}),
                ),
            )
            for i in range(100)
        ]
        results = analyze_batch(pairs)
        assert len(results) == 100

    def test_goodhart_error_propagates_not_partial(self):
        """When one pair in the batch is invalid, the error must propagate"""
        pairs = [
            AnalysisPair(
                observation=make_observation(
                    span_id=f"{i:016x}",
                    observed_fields=frozenset({"a.b"}),
                ),
                claim=make_claim(
                    span_id=f"{i:016x}",
                    claimed_fields=frozenset({"a.b"}),
                ),
            )
            for i in range(3)
        ] + [
            AnalysisPair(observation=None, claim=None),  # invalid pair at index 3
            AnalysisPair(
                observation=make_observation(
                    span_id="ff" * 8,
                    observed_fields=frozenset({"a.b"}),
                ),
                claim=make_claim(
                    span_id="ff" * 8,
                    claimed_fields=frozenset({"a.b"}),
                ),
            ),
        ]
        with pytest.raises(ConsistencyAnalysisError):
            analyze_batch(pairs)

    def test_goodhart_each_finding_has_schema_v1(self):
        """Every finding in a batch result must have schema_version == 1"""
        pairs = [
            AnalysisPair(
                observation=make_observation(
                    span_id=f"{i:016x}",
                    observed_fields=frozenset({f"field.{i}"}),
                ),
                claim=make_claim(
                    span_id=f"{i:016x}",
                    claimed_fields=frozenset({f"field.{i}", f"extra.{i}"}),
                ),
            )
            for i in range(5)
        ]
        results = analyze_batch(pairs)
        assert all(f.schema_version == 1 for f in results)

    def test_goodhart_batch_results_independent(self):
        """Each pair in a batch must be analyzed independently"""
        pairs = [
            AnalysisPair(
                observation=make_observation(
                    node_id="node.alpha",
                    span_id="a1" * 8,
                    observed_fields=frozenset({"x.y"}),
                ),
                claim=None,
            ),
            AnalysisPair(
                observation=make_observation(
                    node_id="node.beta",
                    span_id="b2" * 8,
                    observed_fields=frozenset({"x.y"}),
                ),
                claim=make_claim(
                    node_id="node.beta",
                    span_id="b2" * 8,
                    claimed_fields=frozenset({"x.y"}),
                ),
            ),
        ]
        results = analyze_batch(pairs)
        assert results[0].outcome == ConsistencyOutcome.MISSING_CLAIM
        assert results[1].outcome == ConsistencyOutcome.CONSISTENT
        assert results[0].node_id != results[1].node_id


class TestGoodhartPersistAndQuery:
    """Adversarial tests for persist, get_by_node, get_by_span, has_high_severity."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_path):
        """Ensure a fresh finding store for each test.
        
        Implementations may use different store initialization approaches.
        We try common patterns; if the store uses a file path, we patch it.
        """
        # This fixture may need adaptation per implementation.
        # Attempting common patterns:
        import os
        self._store_path = tmp_path / "findings.jsonl"
        os.environ["FINDINGS_STORE_PATH"] = str(self._store_path)
        yield
        if "FINDINGS_STORE_PATH" in os.environ:
            del os.environ["FINDINGS_STORE_PATH"]

    def test_goodhart_persist_then_get_by_span_multiple(self):
        """Persisting findings for the same span from different analysis runs must all be retrievable via get_by_span"""
        target_span_id = "cc" * 8
        f1 = make_finding(
            span_id=target_span_id,
            analyzed_at="2024-01-15T12:00:00Z",
        )
        f2 = make_finding(
            span_id=target_span_id,
            analyzed_at="2024-01-15T13:00:00Z",
        )
        persist(f1)
        persist(f2)
        results = get_by_span(target_span_id)
        assert len(results) == 2
        assert all(r.span_id == target_span_id for r in results)

    def test_goodhart_get_by_node_no_prefix_match(self):
        """get_by_node must not return findings for other nodes even when those other nodes share similar prefixes"""
        f1 = make_finding(
            node_id="service.auth",
            span_id="d1" * 8,
            analyzed_at="2024-01-15T12:00:00Z",
        )
        f2 = make_finding(
            node_id="service.auth.handler",
            span_id="d2" * 8,
            analyzed_at="2024-01-15T12:00:00Z",
        )
        persist(f1)
        persist(f2)
        results = get_by_node("service.auth")
        assert len(results) == 1
        assert results[0].node_id == "service.auth"

    def test_goodhart_get_by_span_exact_match(self):
        """get_by_span must perform exact span_id matching, not prefix or substring matching"""
        span_a = "aabb" * 4
        span_b = "aabb" * 4 + "cc"
        # Only persist if span_b is actually different (it is, since it's longer)
        f1 = make_finding(span_id=span_a, analyzed_at="2024-01-15T12:00:00Z")
        f2 = make_finding(span_id=span_b, analyzed_at="2024-01-15T12:00:00Z")
        persist(f1)
        persist(f2)
        results = get_by_span(span_a)
        assert len(results) == 1
        assert results[0].span_id == span_a

    def test_goodhart_has_high_severity_exact_boundary(self):
        """has_high_severity with since equal to a HIGH finding's analyzed_at must return True (inclusive >= boundary)"""
        finding = make_finding(
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2024-06-15T10:30:00Z",
        )
        persist(finding)
        result = has_high_severity(
            node_id="test.node", since="2024-06-15T10:30:00Z"
        )
        assert result is True

    def test_goodhart_has_high_severity_one_second_after(self):
        """has_high_severity with since one second after a HIGH finding's analyzed_at must return False"""
        finding = make_finding(
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2024-06-15T10:30:00Z",
        )
        persist(finding)
        result = has_high_severity(
            node_id="test.node", since="2024-06-15T10:30:01Z"
        )
        assert result is False

    def test_goodhart_has_high_severity_multiple_nodes_isolation(self):
        """has_high_severity must only consider findings for the requested node_id"""
        finding = make_finding(
            node_id="other.node",
            severity=FindingSeverity.HIGH,
            outcome=ConsistencyOutcome.MISSING_CLAIM,
            analyzed_at="2024-06-15T10:30:00Z",
        )
        persist(finding)
        result = has_high_severity(node_id="target.node", since=None)
        assert result is False

    def test_goodhart_has_high_severity_medium_not_high(self):
        """MEDIUM severity findings must not trigger has_high_severity"""
        finding = make_finding(
            severity=FindingSeverity.MEDIUM,
            outcome=ConsistencyOutcome.INCONSISTENT,
            analyzed_at="2024-06-15T10:30:00Z",
        )
        persist(finding)
        result = has_high_severity(node_id="test.node", since=None)
        assert result is False

    def test_goodhart_persist_multiple_then_get_by_node_count(self):
        """Persisting N findings for a node and then querying must return exactly N findings"""
        for i in range(5):
            finding = make_finding(
                span_id=f"{i:016x}",
                analyzed_at=f"2024-01-{15+i:02d}T12:00:00Z",
            )
            persist(finding)
        results = get_by_node("test.node")
        assert len(results) == 5

    def test_goodhart_get_by_node_chronological_out_of_order(self):
        """Findings persisted out of chronological order must still be returned sorted by analyzed_at ascending"""
        timestamps = [
            "2024-03-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-02-01T00:00:00Z",
        ]
        for i, ts in enumerate(timestamps):
            finding = make_finding(
                span_id=f"{i:016x}",
                analyzed_at=ts,
            )
            persist(finding)
        results = get_by_node("test.node")
        assert len(results) == 3
        assert results[0].analyzed_at <= results[1].analyzed_at
        assert results[1].analyzed_at <= results[2].analyzed_at
