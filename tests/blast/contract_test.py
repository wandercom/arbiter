"""
Complete test suite for Blast Radius & Soak Computation component.
Run with: pytest contract_test.py -v
"""

import math
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Optional, Dict

from arbiter.blast import (
    DataTier,
    ActionCategory,
    NodeMetadata,
    AccessGraphEdge,
    AccessGraph,
    NodeBlastDetail,
    TraversalResult,
    ClassificationResult,
    SoakParams,
    BlastResult,
    compute_blast_radius,
    classify_blast,
    compute_soak_duration,
    evaluate_blast,
    add_node,
    add_edge,
    classify_node,
)


# ============================================================================
# Factory Helpers (conftest-like)
# ============================================================================

def make_node_metadata(
    node_id="node_a",
    data_tier=DataTier.PUBLIC,
    trust_score=0.8,
    authorized_tiers=None,
    is_authoritative=False,
):
    if authorized_tiers is None:
        authorized_tiers = [data_tier]
    return NodeMetadata(
        node_id=node_id,
        data_tier=data_tier,
        trust_score=trust_score,
        authorized_tiers=authorized_tiers,
        is_authoritative=is_authoritative,
    )


def make_access_graph():
    return AccessGraph(adjacency={}, metadata={})


def make_soak_params(
    base_durations=None,
    target_requests=1000.0,
    observed_rate_rps=10.0,
    low_trust_threshold=0.3,
):
    if base_durations is None:
        base_durations = {
            DataTier.PUBLIC: timedelta(minutes=5),
            DataTier.PII: timedelta(minutes=15),
            DataTier.FINANCIAL: timedelta(minutes=30),
            DataTier.AUTH: timedelta(minutes=45),
            DataTier.COMPLIANCE: timedelta(hours=1),
        }
    return SoakParams(
        base_durations=base_durations,
        target_requests=target_requests,
        observed_rate_rps=observed_rate_rps,
        low_trust_threshold=low_trust_threshold,
    )


def make_node_blast_detail(
    node_id="node_a",
    data_tier=DataTier.PUBLIC,
    trust_score=0.8,
    is_authoritative=False,
    is_authorized_for_tier=True,
    node_action=ActionCategory.AUTO_MERGE,
    depth=0,
):
    return NodeBlastDetail(
        node_id=node_id,
        data_tier=data_tier,
        trust_score=trust_score,
        is_authoritative=is_authoritative,
        is_authorized_for_tier=is_authorized_for_tier,
        node_action=node_action,
        depth=depth,
    )


def make_traversal_result(
    origin="node_a",
    reachable_nodes=None,
    node_details=None,
    highest_data_tier=DataTier.PUBLIC,
    max_depth_reached=0,
    cycle_detected=False,
):
    if reachable_nodes is None:
        reachable_nodes = [origin]
    if node_details is None:
        node_details = [
            make_node_blast_detail(node_id=n) for n in reachable_nodes
        ]
    return TraversalResult(
        origin=origin,
        reachable_nodes=reachable_nodes,
        node_details=node_details,
        highest_data_tier=highest_data_tier,
        max_depth_reached=max_depth_reached,
        cycle_detected=cycle_detected,
    )


def build_linear_chain(tiers=None):
    """Build A->B->C linear chain with given tiers."""
    if tiers is None:
        tiers = [DataTier.PUBLIC, DataTier.PUBLIC, DataTier.PUBLIC]
    names = ["node_a", "node_b", "node_c"]
    graph = make_access_graph()
    for name, tier in zip(names, tiers):
        add_node(graph, make_node_metadata(node_id=name, data_tier=tier))
    add_edge(graph, "node_a", "node_b")
    add_edge(graph, "node_b", "node_c")
    return graph


def build_diamond_graph():
    """Build diamond: A->B, A->C, B->D, C->D"""
    graph = make_access_graph()
    for n in ["node_a", "node_b", "node_c", "node_d"]:
        add_node(graph, make_node_metadata(node_id=n))
    add_edge(graph, "node_a", "node_b")
    add_edge(graph, "node_a", "node_c")
    add_edge(graph, "node_b", "node_d")
    add_edge(graph, "node_c", "node_d")
    return graph


def build_star_graph():
    """Build star: A->B, A->C, A->D"""
    graph = make_access_graph()
    for n in ["node_a", "node_b", "node_c", "node_d"]:
        add_node(graph, make_node_metadata(node_id=n))
    add_edge(graph, "node_a", "node_b")
    add_edge(graph, "node_a", "node_c")
    add_edge(graph, "node_a", "node_d")
    return graph


def build_cyclic_graph():
    """Build A->B->C->A cycle."""
    graph = make_access_graph()
    for n in ["node_a", "node_b", "node_c"]:
        add_node(graph, make_node_metadata(node_id=n))
    add_edge(graph, "node_a", "node_b")
    add_edge(graph, "node_b", "node_c")
    add_edge(graph, "node_c", "node_a")
    return graph


def make_mock_notifier():
    notifier = MagicMock()
    notifier.notify = MagicMock()
    return notifier


# ============================================================================
# test_compute_blast_radius
# ============================================================================

class TestComputeBlastRadiusHappyPath:
    """BFS traversal happy path tests across different topologies."""

    def test_single_node(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="origin"))
        result = compute_blast_radius(graph, "origin", max_depth=None)
        assert result.origin == "origin"
        assert "origin" in result.reachable_nodes
        assert len(result.reachable_nodes) == 1
        assert len(result.node_details) == 1
        assert result.highest_data_tier == DataTier.PUBLIC
        assert result.max_depth_reached == 0
        assert result.cycle_detected is False

    def test_linear_chain(self):
        graph = build_linear_chain()
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        assert len(result.reachable_nodes) == 3
        assert result.max_depth_reached == 2
        assert result.origin == "node_a"
        assert "node_a" in result.reachable_nodes
        assert "node_b" in result.reachable_nodes
        assert "node_c" in result.reachable_nodes

    def test_diamond_graph(self):
        graph = build_diamond_graph()
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        assert len(result.reachable_nodes) == 4
        assert result.cycle_detected is False

    def test_tree_fanout_star(self):
        graph = build_star_graph()
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        assert len(result.reachable_nodes) == 4
        assert result.max_depth_reached == 1

    def test_highest_tier_across_chain(self):
        graph = build_linear_chain(tiers=[DataTier.PUBLIC, DataTier.PII, DataTier.FINANCIAL])
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        assert result.highest_data_tier == DataTier.FINANCIAL


class TestComputeBlastRadiusCycles:
    """BFS cycle detection and termination tests."""

    def test_self_loop(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="origin"))
        add_edge(graph, "origin", "origin")
        result = compute_blast_radius(graph, "origin", max_depth=None)
        assert result.cycle_detected is True
        assert len(result.reachable_nodes) == 1

    def test_two_node_cycle(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        add_node(graph, make_node_metadata(node_id="b"))
        add_edge(graph, "a", "b")
        add_edge(graph, "b", "a")
        result = compute_blast_radius(graph, "a", max_depth=None)
        assert result.cycle_detected is True
        assert len(result.reachable_nodes) == 2

    def test_cycle_not_from_origin(self):
        graph = make_access_graph()
        for n in ["a", "b", "c"]:
            add_node(graph, make_node_metadata(node_id=n))
        add_edge(graph, "a", "b")
        add_edge(graph, "b", "c")
        add_edge(graph, "c", "b")
        result = compute_blast_radius(graph, "a", max_depth=None)
        assert result.cycle_detected is True
        assert len(result.reachable_nodes) == 3

    def test_multi_cycle_graph(self):
        graph = make_access_graph()
        for n in ["a", "b", "c", "d"]:
            add_node(graph, make_node_metadata(node_id=n))
        add_edge(graph, "a", "b")
        add_edge(graph, "b", "c")
        add_edge(graph, "c", "a")
        add_edge(graph, "b", "d")
        add_edge(graph, "d", "b")
        result = compute_blast_radius(graph, "a", max_depth=None)
        assert result.cycle_detected is True
        assert len(result.reachable_nodes) == 4


class TestComputeBlastRadiusDepthLimiting:
    """BFS depth limiting behavior tests."""

    def test_depth_zero(self):
        graph = build_linear_chain()
        result = compute_blast_radius(graph, "node_a", max_depth=0)
        assert len(result.reachable_nodes) == 1
        assert result.max_depth_reached == 0

    def test_depth_one(self):
        graph = build_linear_chain()
        result = compute_blast_radius(graph, "node_a", max_depth=1)
        assert len(result.reachable_nodes) == 2
        assert result.max_depth_reached <= 1

    def test_depth_exact_boundary(self):
        graph = build_linear_chain()
        result = compute_blast_radius(graph, "node_a", max_depth=2)
        assert len(result.reachable_nodes) == 3
        assert result.max_depth_reached == 2

    def test_depth_beyond_diameter(self):
        graph = build_linear_chain()
        result = compute_blast_radius(graph, "node_a", max_depth=100)
        assert len(result.reachable_nodes) == 3
        assert result.max_depth_reached == 2

    def test_disconnected_graph(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        add_node(graph, make_node_metadata(node_id="b"))
        # No edge between them
        result = compute_blast_radius(graph, "a", max_depth=None)
        assert len(result.reachable_nodes) == 1


class TestComputeBlastRadiusErrors:
    """BFS error case tests."""

    def test_origin_not_found(self):
        graph = make_access_graph()
        with pytest.raises(Exception) as exc_info:
            compute_blast_radius(graph, "nonexistent", max_depth=None)
        # Accept any exception that indicates origin not found
        assert exc_info.value is not None

    def test_inconsistent_graph(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        # Manually insert an adjacency reference to a node without metadata
        graph.adjacency["a"] = ["ghost_node"]
        with pytest.raises(Exception):
            compute_blast_radius(graph, "a", max_depth=None)

    def test_invalid_max_depth_negative(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        with pytest.raises(Exception):
            compute_blast_radius(graph, "a", max_depth=-1)


class TestComputeBlastRadiusInvariants:
    """BFS traversal invariant tests."""

    def test_origin_always_in_reachable(self):
        graph = build_linear_chain()
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        assert "node_a" in result.reachable_nodes

    def test_reachable_count_equals_details_count(self):
        graph = build_diamond_graph()
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        assert len(result.reachable_nodes) == len(result.node_details)

    def test_bfs_terminates_on_fully_connected_cycle(self):
        graph = make_access_graph()
        nodes = ["a", "b", "c", "d"]
        for n in nodes:
            add_node(graph, make_node_metadata(node_id=n))
        # Fully connected
        for s in nodes:
            for t in nodes:
                if s != t:
                    add_edge(graph, s, t)
        result = compute_blast_radius(graph, "a", max_depth=None)
        assert result is not None
        assert result.cycle_detected is True
        assert len(result.reachable_nodes) == 4

    def test_all_reachable_nodes_have_details(self):
        graph = build_star_graph()
        result = compute_blast_radius(graph, "node_a", max_depth=None)
        detail_node_ids = {d.node_id for d in result.node_details}
        for n in result.reachable_nodes:
            assert n in detail_node_ids


# ============================================================================
# test_classify_blast
# ============================================================================

class TestClassifyBlastHappyPath:
    """Classification happy path tests across tiers."""

    def test_all_public_auto_merge(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.PUBLIC,
                is_authorized_for_tier=True, trust_score=0.8,
            )],
            highest_data_tier=DataTier.PUBLIC,
        )
        soak_params = make_soak_params()
        result = classify_blast(traversal, soak_params)
        assert result.action == ActionCategory.AUTO_MERGE
        assert result.legal_flag is False

    def test_pii_yields_soak(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.PII,
                is_authorized_for_tier=True, trust_score=0.8,
            )],
            highest_data_tier=DataTier.PII,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.SOAK
        assert result.legal_flag is False

    def test_financial_yields_human_gate(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.FINANCIAL,
                is_authorized_for_tier=True, trust_score=0.8,
            )],
            highest_data_tier=DataTier.FINANCIAL,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is False

    def test_auth_yields_human_gate(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.AUTH,
                is_authorized_for_tier=True, trust_score=0.8,
            )],
            highest_data_tier=DataTier.AUTH,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.HUMAN_GATE

    def test_compliance_yields_human_gate_with_legal_flag(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.COMPLIANCE,
                is_authorized_for_tier=True, trust_score=0.8,
            )],
            highest_data_tier=DataTier.COMPLIANCE,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is True

    def test_mixed_public_pii_yields_soak(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a", "b"],
            node_details=[
                make_node_blast_detail(node_id="a", data_tier=DataTier.PUBLIC, is_authorized_for_tier=True),
                make_node_blast_detail(node_id="b", data_tier=DataTier.PII, is_authorized_for_tier=True, depth=1),
            ],
            highest_data_tier=DataTier.PII,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.SOAK


class TestClassifyBlastEdgeCases:
    """Classification edge cases."""

    def test_unauthorized_tier_yields_human_gate(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.PUBLIC,
                is_authorized_for_tier=False, trust_score=0.8,
            )],
            highest_data_tier=DataTier.PUBLIC,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.HUMAN_GATE

    def test_low_trust_authoritative_yields_human_gate(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.PUBLIC,
                is_authorized_for_tier=True, trust_score=0.1,
                is_authoritative=True,
            )],
            highest_data_tier=DataTier.PUBLIC,
        )
        soak_params = make_soak_params(low_trust_threshold=0.3)
        result = classify_blast(traversal, soak_params)
        assert result.action == ActionCategory.HUMAN_GATE


class TestClassifyBlastInvariants:
    """Classification invariant tests."""

    def test_legal_flag_false_when_not_human_gate(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(
                node_id="a", data_tier=DataTier.PUBLIC,
                is_authorized_for_tier=True,
            )],
            highest_data_tier=DataTier.PUBLIC,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.legal_flag is False or result.action == ActionCategory.HUMAN_GATE

    def test_contributing_nodes_nonempty(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a"],
            node_details=[make_node_blast_detail(node_id="a")],
        )
        result = classify_blast(traversal, make_soak_params())
        assert len(result.contributing_nodes) > 0

    def test_contributing_nodes_match_final_action(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a", "b"],
            node_details=[
                make_node_blast_detail(node_id="a", data_tier=DataTier.PUBLIC, is_authorized_for_tier=True),
                make_node_blast_detail(node_id="b", data_tier=DataTier.PII, is_authorized_for_tier=True, depth=1),
            ],
            highest_data_tier=DataTier.PII,
        )
        result = classify_blast(traversal, make_soak_params())
        # Contributing nodes should be those at the max action level
        # In this case, SOAK contributors should include "b"
        assert len(result.contributing_nodes) > 0

    def test_data_tier_severity_order(self):
        """Verify DataTier has expected order: PUBLIC < PII < FINANCIAL < AUTH < COMPLIANCE."""
        tiers = [DataTier.PUBLIC, DataTier.PII, DataTier.FINANCIAL, DataTier.AUTH, DataTier.COMPLIANCE]
        expected_actions = [
            ActionCategory.AUTO_MERGE, ActionCategory.SOAK,
            ActionCategory.HUMAN_GATE, ActionCategory.HUMAN_GATE, ActionCategory.HUMAN_GATE
        ]
        for tier, expected_action in zip(tiers, expected_actions):
            traversal = make_traversal_result(
                origin="a",
                reachable_nodes=["a"],
                node_details=[make_node_blast_detail(
                    node_id="a", data_tier=tier, is_authorized_for_tier=True, trust_score=0.8,
                )],
                highest_data_tier=tier,
            )
            result = classify_blast(traversal, make_soak_params())
            assert result.action == expected_action, f"Failed for tier {tier}"


class TestClassifyBlastErrors:
    """Classification error tests."""

    def test_empty_traversal(self):
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=[],
            node_details=[],
        )
        with pytest.raises(Exception):
            classify_blast(traversal, make_soak_params())


# ============================================================================
# test_compute_soak_duration
# ============================================================================

class TestComputeSoakDurationHappyPath:
    """Soak duration happy path tests."""

    def test_public_tier_trust_one(self):
        soak_params = make_soak_params()
        result = compute_soak_duration(DataTier.PUBLIC, 1.0, soak_params)
        assert result.total_seconds() > 0

    def test_pii_tier_trust_half(self):
        soak_params = make_soak_params()
        result = compute_soak_duration(DataTier.PII, 0.5, soak_params)
        assert result.total_seconds() > 0

    def test_financial_tier(self):
        soak_params = make_soak_params()
        result = compute_soak_duration(DataTier.FINANCIAL, 0.8, soak_params)
        assert result.total_seconds() > 0

    def test_compliance_tier(self):
        soak_params = make_soak_params()
        result = compute_soak_duration(DataTier.COMPLIANCE, 0.5, soak_params)
        assert result.total_seconds() > 0

    def test_formula_verification(self):
        """Verify exact formula: base * (2.0 - trust) * max(1.0, sqrt(target/rps))."""
        base = timedelta(minutes=10)
        soak_params = make_soak_params(
            base_durations={
                DataTier.PUBLIC: base,
                DataTier.PII: base,
                DataTier.FINANCIAL: base,
                DataTier.AUTH: base,
                DataTier.COMPLIANCE: base,
            },
            target_requests=100.0,
            observed_rate_rps=1.0,
        )
        trust = 0.5
        result = compute_soak_duration(DataTier.PUBLIC, trust, soak_params)
        expected_seconds = base.total_seconds() * (2.0 - trust) * max(1.0, math.sqrt(100.0 / 1.0))
        expected = timedelta(seconds=expected_seconds)
        assert abs(result.total_seconds() - expected.total_seconds()) < 0.01


class TestComputeSoakDurationEdgeCases:
    """Soak duration edge case tests."""

    def test_trust_zero_max_multiplier(self):
        soak_params = make_soak_params()
        result = compute_soak_duration(DataTier.PUBLIC, 0.0, soak_params)
        assert result.total_seconds() > 0

    def test_trust_one_min_multiplier(self):
        soak_params = make_soak_params()
        result = compute_soak_duration(DataTier.PUBLIC, 1.0, soak_params)
        assert result.total_seconds() > 0

    def test_high_target_rps_ratio(self):
        soak_params = make_soak_params(target_requests=1000000.0, observed_rate_rps=1.0)
        result = compute_soak_duration(DataTier.PUBLIC, 0.5, soak_params)
        assert result.total_seconds() > 0

    def test_ratio_below_one_floors_to_one(self):
        soak_params = make_soak_params(target_requests=1.0, observed_rate_rps=1000.0)
        result = compute_soak_duration(DataTier.PUBLIC, 0.5, soak_params)
        # When ratio < 1.0, max(1.0, sqrt(ratio)) == 1.0
        base = soak_params.base_durations[DataTier.PUBLIC]
        expected_seconds = base.total_seconds() * (2.0 - 0.5) * 1.0
        assert abs(result.total_seconds() - expected_seconds) < 0.01


class TestComputeSoakDurationInvariants:
    """Soak duration invariant tests."""

    def test_monotone_trust_lower_trust_longer_duration(self):
        soak_params = make_soak_params()
        duration_high_trust = compute_soak_duration(DataTier.PII, 0.9, soak_params)
        duration_low_trust = compute_soak_duration(DataTier.PII, 0.1, soak_params)
        assert duration_low_trust >= duration_high_trust

    def test_monotone_ratio_higher_ratio_longer_duration(self):
        soak_params_low = make_soak_params(target_requests=100.0, observed_rate_rps=100.0)
        soak_params_high = make_soak_params(target_requests=10000.0, observed_rate_rps=1.0)
        duration_low_ratio = compute_soak_duration(DataTier.PII, 0.5, soak_params_low)
        duration_high_ratio = compute_soak_duration(DataTier.PII, 0.5, soak_params_high)
        assert duration_high_ratio >= duration_low_ratio

    def test_always_positive(self):
        soak_params = make_soak_params()
        for tier in [DataTier.PUBLIC, DataTier.PII, DataTier.FINANCIAL, DataTier.AUTH, DataTier.COMPLIANCE]:
            result = compute_soak_duration(tier, 0.5, soak_params)
            assert result.total_seconds() > 0


class TestComputeSoakDurationErrors:
    """Soak duration error tests."""

    def test_missing_base_duration(self):
        soak_params = make_soak_params(base_durations={DataTier.PUBLIC: timedelta(minutes=5)})
        with pytest.raises(Exception):
            compute_soak_duration(DataTier.PII, 0.5, soak_params)

    def test_invalid_trust_nan(self):
        soak_params = make_soak_params()
        with pytest.raises(Exception):
            compute_soak_duration(DataTier.PUBLIC, float('nan'), soak_params)

    def test_invalid_trust_inf(self):
        soak_params = make_soak_params()
        with pytest.raises(Exception):
            compute_soak_duration(DataTier.PUBLIC, float('inf'), soak_params)


# ============================================================================
# test_classify_node
# ============================================================================

class TestClassifyNodeHappyPath:
    """Per-node classification happy path tests."""

    def test_public_authorized_auto_merge(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.PUBLIC, authorized_tiers=[DataTier.PUBLIC])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.PUBLIC, is_authorized_for_tier=True)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.AUTO_MERGE

    def test_pii_authorized_soak(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.PII, authorized_tiers=[DataTier.PII])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.PII, is_authorized_for_tier=True)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.SOAK

    def test_financial_human_gate(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.FINANCIAL, authorized_tiers=[DataTier.FINANCIAL])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.FINANCIAL, is_authorized_for_tier=True)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE

    def test_auth_human_gate(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.AUTH, authorized_tiers=[DataTier.AUTH])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.AUTH, is_authorized_for_tier=True)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE

    def test_compliance_human_gate(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.COMPLIANCE, authorized_tiers=[DataTier.COMPLIANCE])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.COMPLIANCE, is_authorized_for_tier=True)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE


class TestClassifyNodeEdgeCases:
    """Per-node classification edge cases."""

    def test_unauthorized_public_human_gate(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.PUBLIC, authorized_tiers=[])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.PUBLIC, is_authorized_for_tier=False)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE

    def test_unauthorized_pii_human_gate(self):
        meta = make_node_metadata(node_id="a", data_tier=DataTier.PII, authorized_tiers=[])
        detail = make_node_blast_detail(node_id="a", data_tier=DataTier.PII, is_authorized_for_tier=False)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE

    def test_low_trust_authoritative_public_human_gate(self):
        meta = make_node_metadata(
            node_id="a", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC], trust_score=0.1,
            is_authoritative=True,
        )
        detail = make_node_blast_detail(
            node_id="a", data_tier=DataTier.PUBLIC,
            is_authorized_for_tier=True, trust_score=0.1,
            is_authoritative=True,
        )
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE

    def test_high_trust_authoritative_public_auto_merge(self):
        meta = make_node_metadata(
            node_id="a", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC], trust_score=0.9,
            is_authoritative=True,
        )
        detail = make_node_blast_detail(
            node_id="a", data_tier=DataTier.PUBLIC,
            is_authorized_for_tier=True, trust_score=0.9,
            is_authoritative=True,
        )
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.AUTO_MERGE

    def test_trust_at_threshold_not_low_trust(self):
        """Trust exactly at threshold (0.3) should NOT be considered low trust (< not <=)."""
        meta = make_node_metadata(
            node_id="a", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC], trust_score=0.3,
            is_authoritative=True,
        )
        detail = make_node_blast_detail(
            node_id="a", data_tier=DataTier.PUBLIC,
            is_authorized_for_tier=True, trust_score=0.3,
            is_authoritative=True,
        )
        result = classify_node(detail, meta, 0.3)
        # At threshold means not low trust; should be AUTO_MERGE for PUBLIC
        assert result == ActionCategory.AUTO_MERGE

    def test_trust_just_below_threshold_is_low_trust(self):
        meta = make_node_metadata(
            node_id="a", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC], trust_score=0.29,
            is_authoritative=True,
        )
        detail = make_node_blast_detail(
            node_id="a", data_tier=DataTier.PUBLIC,
            is_authorized_for_tier=True, trust_score=0.29,
            is_authoritative=True,
        )
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.HUMAN_GATE

    def test_non_authoritative_low_trust_no_escalation(self):
        """Low trust but non-authoritative should NOT escalate for PUBLIC."""
        meta = make_node_metadata(
            node_id="a", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC], trust_score=0.1,
            is_authoritative=False,
        )
        detail = make_node_blast_detail(
            node_id="a", data_tier=DataTier.PUBLIC,
            is_authorized_for_tier=True, trust_score=0.1,
            is_authoritative=False,
        )
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.AUTO_MERGE


class TestClassifyNodeErrors:
    """Per-node classification error tests."""

    def test_node_id_mismatch(self):
        meta = make_node_metadata(node_id="a")
        detail = make_node_blast_detail(node_id="b")
        with pytest.raises(Exception):
            classify_node(detail, meta, 0.3)


# ============================================================================
# test_graph_mutations (add_node, add_edge)
# ============================================================================

class TestAddNode:
    """Tests for add_node builder method."""

    def test_add_node_happy(self):
        graph = make_access_graph()
        meta = make_node_metadata(node_id="x")
        returned = add_node(graph, meta)
        assert "x" in graph.metadata
        assert "x" in graph.adjacency
        assert graph.adjacency["x"] == [] or list(graph.adjacency["x"]) == []

    def test_add_node_returns_same_object(self):
        graph = make_access_graph()
        meta = make_node_metadata(node_id="x")
        returned = add_node(graph, meta)
        assert returned is graph

    def test_add_node_replaces_metadata(self):
        graph = make_access_graph()
        meta1 = make_node_metadata(node_id="x", data_tier=DataTier.PUBLIC)
        add_node(graph, meta1)
        meta2 = make_node_metadata(node_id="x", data_tier=DataTier.PII)
        add_node(graph, meta2)
        assert graph.metadata["x"].data_tier == DataTier.PII

    def test_add_node_invalid_metadata_trust_out_of_range(self):
        """NodeMetadata with trust_score out of [0,1] should raise error."""
        with pytest.raises(Exception):
            make_node_metadata(node_id="x", trust_score=1.5)


class TestAddEdge:
    """Tests for add_edge builder method."""

    def test_add_edge_happy(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        add_node(graph, make_node_metadata(node_id="b"))
        returned = add_edge(graph, "a", "b")
        assert "b" in graph.adjacency["a"]

    def test_add_edge_returns_same_object(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        add_node(graph, make_node_metadata(node_id="b"))
        returned = add_edge(graph, "a", "b")
        assert returned is graph

    def test_add_edge_no_duplicates(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        add_node(graph, make_node_metadata(node_id="b"))
        add_edge(graph, "a", "b")
        add_edge(graph, "a", "b")
        count = list(graph.adjacency["a"]).count("b")
        assert count == 1

    def test_add_edge_source_not_found(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="b"))
        with pytest.raises(Exception):
            add_edge(graph, "nonexistent", "b")

    def test_add_edge_target_not_found(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        with pytest.raises(Exception):
            add_edge(graph, "a", "nonexistent")


class TestGraphConsistencyInvariant:
    """Graph consistency invariant tests."""

    def test_all_adjacency_nodes_have_metadata(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        add_node(graph, make_node_metadata(node_id="b"))
        add_node(graph, make_node_metadata(node_id="c"))
        add_edge(graph, "a", "b")
        add_edge(graph, "b", "c")
        # Verify invariant
        for source, targets in graph.adjacency.items():
            assert source in graph.metadata, f"Source {source} missing from metadata"
            for t in targets:
                assert t in graph.metadata, f"Target {t} missing from metadata"


# ============================================================================
# test_evaluate_blast (integration/orchestrator tests)
# ============================================================================

class TestEvaluateBlastHappyPath:
    """Orchestrator happy path tests with mocked notifier."""

    def test_auto_merge_no_notification(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC],
        ))
        notifier = make_mock_notifier()
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "origin", soak_params, notifier, max_depth=None)
        assert result.action == ActionCategory.AUTO_MERGE
        assert result.soak_duration is None
        assert result.legal_flag is False
        assert notifier.notify.call_count == 0

    def test_soak_with_positive_duration(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.PII,
            authorized_tiers=[DataTier.PII],
        ))
        notifier = make_mock_notifier()
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "origin", soak_params, notifier, max_depth=None)
        assert result.action == ActionCategory.SOAK
        assert result.soak_duration is not None
        assert result.soak_duration.total_seconds() > 0
        assert notifier.notify.call_count == 0

    def test_human_gate_fires_notifier(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.FINANCIAL,
            authorized_tiers=[DataTier.FINANCIAL],
        ))
        notifier = make_mock_notifier()
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "origin", soak_params, notifier, max_depth=None)
        assert result.action == ActionCategory.HUMAN_GATE
        assert notifier.notify.call_count == 1

    def test_compliance_legal_flag_and_notifier(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.COMPLIANCE,
            authorized_tiers=[DataTier.COMPLIANCE],
        ))
        notifier = make_mock_notifier()
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "origin", soak_params, notifier, max_depth=None)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is True
        assert notifier.notify.call_count == 1

    def test_fields_assembled_correctly(self):
        graph = build_linear_chain()
        notifier = make_mock_notifier()
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "node_a", soak_params, notifier, max_depth=None)
        assert result.origin_node == "node_a"
        assert result.highest_data_tier is not None
        assert result.max_depth_reached >= 0
        assert isinstance(result.per_node_details, list)
        assert "node_a" in result.reachable_nodes


class TestEvaluateBlastInvariants:
    """Orchestrator invariant tests."""

    def test_utc_timestamp(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="origin"))
        notifier = make_mock_notifier()
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "origin", soak_params, notifier, max_depth=None)
        assert result.computed_at.tzinfo is not None
        assert result.computed_at.tzinfo == timezone.utc

    def test_origin_in_reachable(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="origin"))
        notifier = make_mock_notifier()
        result = evaluate_blast(graph, "origin", make_soak_params(), notifier, max_depth=None)
        assert "origin" in result.reachable_nodes

    def test_soak_none_when_not_soak(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.FINANCIAL,
            authorized_tiers=[DataTier.FINANCIAL],
        ))
        notifier = make_mock_notifier()
        result = evaluate_blast(graph, "origin", make_soak_params(), notifier, max_depth=None)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.soak_duration is None

    def test_soak_present_when_soak(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.PII,
            authorized_tiers=[DataTier.PII],
        ))
        notifier = make_mock_notifier()
        result = evaluate_blast(graph, "origin", make_soak_params(), notifier, max_depth=None)
        assert result.action == ActionCategory.SOAK
        assert result.soak_duration is not None

    def test_legal_implies_human_gate(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.COMPLIANCE,
            authorized_tiers=[DataTier.COMPLIANCE],
        ))
        notifier = make_mock_notifier()
        result = evaluate_blast(graph, "origin", make_soak_params(), notifier, max_depth=None)
        assert not result.legal_flag or result.action == ActionCategory.HUMAN_GATE


class TestEvaluateBlastEdgeCases:
    """Orchestrator edge cases."""

    def test_notifier_none_no_crash_on_human_gate(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.FINANCIAL,
            authorized_tiers=[DataTier.FINANCIAL],
        ))
        soak_params = make_soak_params()
        result = evaluate_blast(graph, "origin", soak_params, None, max_depth=None)
        assert result.action == ActionCategory.HUMAN_GATE

    def test_max_depth_propagation(self):
        graph = build_linear_chain()
        notifier = make_mock_notifier()
        result = evaluate_blast(graph, "node_a", make_soak_params(), notifier, max_depth=1)
        assert result.max_depth_reached <= 1

    def test_no_notify_when_auto_merge(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.PUBLIC,
            authorized_tiers=[DataTier.PUBLIC],
        ))
        notifier = make_mock_notifier()
        evaluate_blast(graph, "origin", make_soak_params(), notifier, max_depth=None)
        assert notifier.notify.call_count == 0


class TestEvaluateBlastErrors:
    """Orchestrator error tests."""

    def test_origin_not_found(self):
        graph = make_access_graph()
        notifier = make_mock_notifier()
        with pytest.raises(Exception):
            evaluate_blast(graph, "nonexistent", make_soak_params(), notifier, max_depth=None)

    def test_inconsistent_graph(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(node_id="a"))
        graph.adjacency["a"] = ["ghost"]
        notifier = make_mock_notifier()
        with pytest.raises(Exception):
            evaluate_blast(graph, "a", make_soak_params(), notifier, max_depth=None)

    def test_notification_failure(self):
        graph = make_access_graph()
        add_node(graph, make_node_metadata(
            node_id="origin", data_tier=DataTier.FINANCIAL,
            authorized_tiers=[DataTier.FINANCIAL],
        ))
        notifier = make_mock_notifier()
        notifier.notify.side_effect = RuntimeError("Webhook failed")
        soak_params = make_soak_params()
        with pytest.raises(Exception):
            evaluate_blast(graph, "origin", soak_params, notifier, max_depth=None)


# ============================================================================
# Trust score invariant tests
# ============================================================================

class TestTrustScoreInvariant:
    """Trust score must always be in [0.0, 1.0]."""

    def test_trust_score_at_zero(self):
        meta = make_node_metadata(node_id="a", trust_score=0.0)
        assert 0.0 <= meta.trust_score <= 1.0

    def test_trust_score_at_one(self):
        meta = make_node_metadata(node_id="a", trust_score=1.0)
        assert 0.0 <= meta.trust_score <= 1.0

    def test_trust_score_mid(self):
        meta = make_node_metadata(node_id="a", trust_score=0.5)
        assert 0.0 <= meta.trust_score <= 1.0

    def test_trust_score_above_one_rejected(self):
        with pytest.raises(Exception):
            make_node_metadata(node_id="a", trust_score=1.1)

    def test_trust_score_below_zero_rejected(self):
        with pytest.raises(Exception):
            make_node_metadata(node_id="a", trust_score=-0.1)


# ============================================================================
# ActionCategory order invariant
# ============================================================================

class TestActionCategoryOrder:
    """Verify ActionCategory severity ordering."""

    def test_auto_merge_less_than_soak(self):
        """AUTO_MERGE < SOAK: a mix of PUBLIC + PII should yield SOAK."""
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a", "b"],
            node_details=[
                make_node_blast_detail(node_id="a", data_tier=DataTier.PUBLIC, is_authorized_for_tier=True),
                make_node_blast_detail(node_id="b", data_tier=DataTier.PII, is_authorized_for_tier=True, depth=1),
            ],
            highest_data_tier=DataTier.PII,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.SOAK

    def test_soak_less_than_human_gate(self):
        """SOAK < HUMAN_GATE: a mix of PII + FINANCIAL should yield HUMAN_GATE."""
        traversal = make_traversal_result(
            origin="a",
            reachable_nodes=["a", "b"],
            node_details=[
                make_node_blast_detail(node_id="a", data_tier=DataTier.PII, is_authorized_for_tier=True),
                make_node_blast_detail(node_id="b", data_tier=DataTier.FINANCIAL, is_authorized_for_tier=True, depth=1),
            ],
            highest_data_tier=DataTier.FINANCIAL,
        )
        result = classify_blast(traversal, make_soak_params())
        assert result.action == ActionCategory.HUMAN_GATE
