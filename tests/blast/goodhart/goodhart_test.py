"""
Adversarial hidden acceptance tests for Blast Radius & Soak Computation.
These tests catch implementations that pass visible tests via shortcuts/hardcoding.
"""
import math
import pytest
from datetime import timedelta, datetime, timezone
from unittest.mock import MagicMock, call
from arbiter.blast import (
    DataTier, ActionCategory, NodeId, NodeMetadata, AccessGraph,
    AccessGraphEdge, NodeBlastDetail, TraversalResult, ClassificationResult,
    SoakParams, BlastResult, 
    compute_blast_radius, classify_blast, compute_soak_duration,
    evaluate_blast, add_node, add_edge, classify_node,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def make_node(node_id: str, data_tier=DataTier.PUBLIC, trust_score=0.8,
              authorized_tiers=None, is_authoritative=False) -> NodeMetadata:
    if authorized_tiers is None:
        authorized_tiers = [data_tier]
    return NodeMetadata(
        node_id=NodeId(node_id),
        data_tier=data_tier,
        trust_score=trust_score,
        authorized_tiers=authorized_tiers,
        is_authoritative=is_authoritative,
    )


def make_graph(nodes, edges=None):
    g = AccessGraph(adjacency={}, metadata={})
    for n in nodes:
        add_node(g, n)
    for src, tgt in (edges or []):
        add_edge(g, NodeId(src), NodeId(tgt))
    return g


def default_soak_params(low_trust_threshold=0.5, target_requests=100.0,
                         observed_rate_rps=10.0, base_override=None):
    base = base_override or {
        DataTier.PUBLIC: timedelta(minutes=5),
        DataTier.PII: timedelta(minutes=30),
        DataTier.FINANCIAL: timedelta(hours=1),
        DataTier.AUTH: timedelta(hours=2),
        DataTier.COMPLIANCE: timedelta(hours=4),
    }
    return SoakParams(
        base_durations=base,
        target_requests=target_requests,
        observed_rate_rps=observed_rate_rps,
        low_trust_threshold=low_trust_threshold,
    )


def make_detail(node_id, data_tier, trust_score, is_authoritative, is_authorized, action, depth):
    return NodeBlastDetail(
        node_id=NodeId(node_id),
        data_tier=data_tier,
        trust_score=trust_score,
        is_authoritative=is_authoritative,
        is_authorized_for_tier=is_authorized,
        node_action=action,
        depth=depth,
    )


def make_traversal(origin, nodes_info):
    """nodes_info: list of (id, DataTier, trust, is_authoritative, authorized_tiers, depth)"""
    reachable = [NodeId(n[0]) for n in nodes_info]
    details = []
    highest = DataTier.PUBLIC
    max_d = 0
    for nid, tier, trust, auth, auth_tiers, depth in nodes_info:
        is_auth_for_tier = tier in auth_tiers
        details.append(NodeBlastDetail(
            node_id=NodeId(nid),
            data_tier=tier,
            trust_score=trust,
            is_authoritative=auth,
            is_authorized_for_tier=is_auth_for_tier,
            node_action=ActionCategory.AUTO_MERGE,  # placeholder
            depth=depth,
        ))
        if _tier_order(tier) > _tier_order(highest):
            highest = tier
        if depth > max_d:
            max_d = depth
    return TraversalResult(
        origin=NodeId(origin),
        reachable_nodes=reachable,
        node_details=details,
        highest_data_tier=highest,
        max_depth_reached=max_d,
        cycle_detected=False,
    )


def _tier_order(t):
    return [DataTier.PUBLIC, DataTier.PII, DataTier.FINANCIAL, DataTier.AUTH, DataTier.COMPLIANCE].index(t)


# ── compute_blast_radius tests ──────────────────────────────────────────

class TestGoodhartComputeBlastRadius:

    def test_goodhart_cbr_depth_two_excludes_depth_three(self):
        """BFS with max_depth=2 on a longer chain must correctly stop at depth 2, excluding deeper nodes."""
        nodes = [make_node(c) for c in "ABCDE"]
        g = make_graph(nodes, [("A","B"),("B","C"),("C","D"),("D","E")])
        result = compute_blast_radius(g, NodeId("A"), max_depth=2)
        assert len(result.reachable_nodes) == 3
        assert result.max_depth_reached == 2
        assert NodeId("D") not in result.reachable_nodes
        assert NodeId("E") not in result.reachable_nodes

    def test_goodhart_cbr_node_details_depth_correct(self):
        """Each node_detail must carry the correct BFS depth from the origin, not a constant."""
        nodes = [make_node("A"), make_node("B"), make_node("C")]
        g = make_graph(nodes, [("A","B"),("B","C")])
        result = compute_blast_radius(g, NodeId("A"), max_depth=None)
        depth_map = {d.node_id: d.depth for d in result.node_details}
        assert depth_map[NodeId("A")] == 0
        assert depth_map[NodeId("B")] == 1
        assert depth_map[NodeId("C")] == 2

    def test_goodhart_cbr_node_details_carry_metadata(self):
        """Node details must reflect actual per-node metadata, not hardcoded defaults."""
        a = make_node("A", DataTier.PII, 0.3, [DataTier.PII], True)
        b = make_node("B", DataTier.FINANCIAL, 0.9, [DataTier.FINANCIAL], False)
        g = make_graph([a, b], [("A","B")])
        result = compute_blast_radius(g, NodeId("A"), max_depth=None)
        detail_map = {d.node_id: d for d in result.node_details}
        da = detail_map[NodeId("A")]
        db = detail_map[NodeId("B")]
        assert da.data_tier == DataTier.PII
        assert da.trust_score == pytest.approx(0.3)
        assert da.is_authoritative is True
        assert db.data_tier == DataTier.FINANCIAL
        assert db.trust_score == pytest.approx(0.9)
        assert db.is_authoritative is False

    def test_goodhart_cbr_highest_tier_auth(self):
        """highest_data_tier must correctly resolve AUTH as the maximum when mixed with PUBLIC and PII."""
        nodes = [
            make_node("A", DataTier.PUBLIC),
            make_node("B", DataTier.PII),
            make_node("C", DataTier.AUTH),
        ]
        g = make_graph(nodes, [("A","B"),("B","C")])
        result = compute_blast_radius(g, NodeId("A"), max_depth=None)
        assert result.highest_data_tier == DataTier.AUTH

    def test_goodhart_cbr_cycle_detected_false_no_cycle(self):
        """cycle_detected must be False on acyclic graphs, ensuring it's not always True."""
        nodes = [make_node("A"), make_node("B"), make_node("C")]
        g = make_graph(nodes, [("A","B"),("B","C")])
        result = compute_blast_radius(g, NodeId("A"), max_depth=None)
        assert result.cycle_detected is False

    def test_goodhart_cbr_large_cycle_ring(self):
        """BFS on a 5-node ring must terminate, detect cycle, and visit all 5 nodes."""
        nodes = [make_node(c) for c in "ABCDE"]
        g = make_graph(nodes, [("A","B"),("B","C"),("C","D"),("D","E"),("E","A")])
        result = compute_blast_radius(g, NodeId("A"), max_depth=None)
        assert len(result.reachable_nodes) == 5
        assert result.cycle_detected is True
        assert len(result.node_details) == 5

    def test_goodhart_cbr_max_depth_none_unlimited(self):
        """When max_depth is None, BFS traverses the entire reachable graph without depth limit."""
        nodes = [make_node(c) for c in "ABCDEF"]
        edges = [("A","B"),("B","C"),("C","D"),("D","E"),("E","F")]
        g = make_graph(nodes, edges)
        result = compute_blast_radius(g, NodeId("A"), max_depth=None)
        assert len(result.reachable_nodes) == 6
        assert result.max_depth_reached == 5

    def test_goodhart_cbr_directed_edge_not_reverse(self):
        """BFS respects edge direction: A->B does NOT make A reachable from B."""
        nodes = [make_node("A"), make_node("B")]
        g = make_graph(nodes, [("A","B")])
        result = compute_blast_radius(g, NodeId("B"), max_depth=None)
        assert len(result.reachable_nodes) == 1
        assert NodeId("B") in result.reachable_nodes
        assert NodeId("A") not in result.reachable_nodes

    def test_goodhart_cbr_origin_depth_zero(self):
        """The origin node must always be at BFS depth 0."""
        nodes = [make_node("X"), make_node("Y"), make_node("Z")]
        g = make_graph(nodes, [("X","Y"),("Y","Z")])
        result = compute_blast_radius(g, NodeId("X"), max_depth=None)
        detail_map = {d.node_id: d for d in result.node_details}
        assert detail_map[NodeId("X")].depth == 0

    def test_goodhart_cbr_branching_with_depth_limit(self):
        """max_depth limits all branches equally, not just the first path."""
        nodes = [make_node(c) for c in "ABCDEFG"]
        edges = [("A","B"),("A","C"),("B","D"),("C","E"),("D","F"),("E","G")]
        g = make_graph(nodes, edges)
        result = compute_blast_radius(g, NodeId("A"), max_depth=2)
        ids = set(result.reachable_nodes)
        assert NodeId("A") in ids
        assert NodeId("B") in ids
        assert NodeId("C") in ids
        assert NodeId("D") in ids
        assert NodeId("E") in ids
        assert NodeId("F") not in ids
        assert NodeId("G") not in ids
        assert len(result.reachable_nodes) == 5

    def test_goodhart_cbr_inconsistent_graph_in_values(self):
        """Inconsistency in adjacency values (neighbor without metadata) must raise error."""
        g = AccessGraph(
            adjacency={NodeId("A"): [NodeId("B")]},
            metadata={NodeId("A"): make_node("A")},
        )
        with pytest.raises(Exception):
            compute_blast_radius(g, NodeId("A"), max_depth=None)


# ── classify_blast tests ────────────────────────────────────────────────

class TestGoodhartClassifyBlast:

    def test_goodhart_cb_pii_plus_financial_human_gate(self):
        """Mixing PII and FINANCIAL nodes: max-over-nodes yields HUMAN_GATE, legal_flag=False."""
        trav = make_traversal("A", [
            ("A", DataTier.PII, 0.8, False, [DataTier.PII], 0),
            ("B", DataTier.FINANCIAL, 0.8, False, [DataTier.FINANCIAL], 1),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is False

    def test_goodhart_cb_compliance_plus_public_legal_flag(self):
        """PUBLIC + COMPLIANCE mix must yield HUMAN_GATE with legal_flag=True."""
        trav = make_traversal("A", [
            ("A", DataTier.PUBLIC, 0.9, False, [DataTier.PUBLIC], 0),
            ("B", DataTier.COMPLIANCE, 0.9, False, [DataTier.COMPLIANCE], 1),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is True

    def test_goodhart_cb_financial_no_legal_flag(self):
        """HUMAN_GATE from FINANCIAL alone must not set legal_flag (only COMPLIANCE sets it)."""
        trav = make_traversal("A", [
            ("A", DataTier.FINANCIAL, 0.8, False, [DataTier.FINANCIAL], 0),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is False

    def test_goodhart_cb_auth_no_legal_flag(self):
        """HUMAN_GATE from AUTH alone must not set legal_flag."""
        trav = make_traversal("A", [
            ("A", DataTier.AUTH, 0.8, False, [DataTier.AUTH], 0),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is False

    def test_goodhart_cb_unauthorized_pii_human_gate_not_soak(self):
        """PII node NOT authorized for its tier must escalate to HUMAN_GATE, not SOAK."""
        trav = make_traversal("A", [
            ("A", DataTier.PII, 0.9, False, [DataTier.PUBLIC], 0),  # authorized only for PUBLIC
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE

    def test_goodhart_cb_contributing_nodes_only_max_action(self):
        """contributing_nodes must contain only nodes whose per-node action equals the final max action."""
        trav = make_traversal("A", [
            ("A", DataTier.PUBLIC, 0.9, False, [DataTier.PUBLIC], 0),
            ("B", DataTier.PUBLIC, 0.9, False, [DataTier.PUBLIC], 1),
            ("C", DataTier.PUBLIC, 0.9, False, [DataTier.PUBLIC], 1),
            ("D", DataTier.FINANCIAL, 0.9, False, [DataTier.FINANCIAL], 2),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert len(result.contributing_nodes) >= 1
        contributing_ids = {n if isinstance(n, str) else getattr(n, 'node_id', n) for n in result.contributing_nodes}
        # Only D should be contributing (the FINANCIAL node)
        # Allow NodeId or string comparison
        assert NodeId("D") in result.contributing_nodes or "D" in contributing_ids

    def test_goodhart_cb_multiple_contributing_nodes(self):
        """When multiple nodes are at the max severity, all should be in contributing_nodes."""
        trav = make_traversal("A", [
            ("A", DataTier.PUBLIC, 0.9, False, [DataTier.PUBLIC], 0),
            ("B", DataTier.FINANCIAL, 0.9, False, [DataTier.FINANCIAL], 1),
            ("C", DataTier.FINANCIAL, 0.9, False, [DataTier.FINANCIAL], 1),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert len(result.contributing_nodes) >= 2

    def test_goodhart_cb_low_trust_authoritative_pii_escalation(self):
        """Authoritative PII node with low trust escalates to HUMAN_GATE even though PII normally yields SOAK."""
        trav = make_traversal("A", [
            ("A", DataTier.PII, 0.1, True, [DataTier.PII], 0),
        ])
        sp = default_soak_params(low_trust_threshold=0.5)
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE

    def test_goodhart_cb_single_compliance_only(self):
        """Single COMPLIANCE node yields HUMAN_GATE, legal_flag=True, one contributing node."""
        trav = make_traversal("A", [
            ("A", DataTier.COMPLIANCE, 0.9, False, [DataTier.COMPLIANCE], 0),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.HUMAN_GATE
        assert result.legal_flag is True
        assert len(result.contributing_nodes) == 1

    def test_goodhart_cb_three_pii_all_authorized(self):
        """Multiple PII authorized nodes without higher tiers should yield SOAK, not escalate."""
        trav = make_traversal("A", [
            ("A", DataTier.PII, 0.8, False, [DataTier.PII], 0),
            ("B", DataTier.PII, 0.7, False, [DataTier.PII], 1),
            ("C", DataTier.PII, 0.6, False, [DataTier.PII], 2),
        ])
        sp = default_soak_params()
        result = classify_blast(trav, sp)
        assert result.action == ActionCategory.SOAK
        assert result.legal_flag is False
        assert len(result.contributing_nodes) == 3


# ── compute_soak_duration tests ─────────────────────────────────────────

class TestGoodhartComputeSoakDuration:

    def test_goodhart_csd_trust_midpoint_formula(self):
        """Soak with trust=0.75, ratio=1.0 must match formula: base*(2.0-0.75)*1.0."""
        sp = SoakParams(
            base_durations={DataTier.PII: timedelta(hours=1)},
            target_requests=100.0,
            observed_rate_rps=100.0,
            low_trust_threshold=0.5,
        )
        result = compute_soak_duration(DataTier.PII, 0.75, sp)
        expected = timedelta(hours=1) * 1.25 * 1.0
        assert result == expected

    def test_goodhart_csd_ratio_exactly_one(self):
        """When target==rps, ratio factor is 1.0, so only trust multiplier applies."""
        sp = SoakParams(
            base_durations={DataTier.PUBLIC: timedelta(minutes=30)},
            target_requests=50.0,
            observed_rate_rps=50.0,
            low_trust_threshold=0.5,
        )
        result = compute_soak_duration(DataTier.PUBLIC, 0.5, sp)
        expected = timedelta(minutes=30) * 1.5 * 1.0
        assert result == expected

    def test_goodhart_csd_very_small_rps_epsilon_floor(self):
        """Extremely small rps should be floored to epsilon, yielding large but finite duration."""
        sp = SoakParams(
            base_durations={DataTier.PUBLIC: timedelta(minutes=1)},
            target_requests=100.0,
            observed_rate_rps=1e-10,
            low_trust_threshold=0.5,
        )
        result = compute_soak_duration(DataTier.PUBLIC, 0.5, sp)
        assert result > timedelta(0)
        assert result > timedelta(minutes=1)
        # Should be finite
        assert result.total_seconds() < float('inf')

    def test_goodhart_csd_different_base_durations(self):
        """Different tier base durations must produce proportionally different soak durations."""
        sp = SoakParams(
            base_durations={
                DataTier.PII: timedelta(hours=2),
                DataTier.FINANCIAL: timedelta(hours=4),
            },
            target_requests=100.0,
            observed_rate_rps=100.0,
            low_trust_threshold=0.5,
        )
        r_pii = compute_soak_duration(DataTier.PII, 0.5, sp)
        r_fin = compute_soak_duration(DataTier.FINANCIAL, 0.5, sp)
        assert r_fin == r_pii * 2

    def test_goodhart_csd_negative_trust_inf(self):
        """Negative infinity trust_score must be rejected."""
        sp = SoakParams(
            base_durations={DataTier.PUBLIC: timedelta(minutes=5)},
            target_requests=100.0,
            observed_rate_rps=10.0,
            low_trust_threshold=0.5,
        )
        with pytest.raises(Exception):
            compute_soak_duration(DataTier.PUBLIC, float('-inf'), sp)

    def test_goodhart_csd_auth_tier_formula(self):
        """Soak for AUTH tier must use AUTH's base_duration specifically."""
        sp = SoakParams(
            base_durations={
                DataTier.PUBLIC: timedelta(minutes=5),
                DataTier.PII: timedelta(minutes=30),
                DataTier.FINANCIAL: timedelta(hours=1),
                DataTier.AUTH: timedelta(hours=2),
                DataTier.COMPLIANCE: timedelta(hours=4),
            },
            target_requests=100.0,
            observed_rate_rps=100.0,
            low_trust_threshold=0.5,
        )
        result = compute_soak_duration(DataTier.AUTH, 0.5, sp)
        expected = timedelta(hours=2) * 1.5 * 1.0
        assert result == expected

    def test_goodhart_csd_ratio_four_sqrt_two(self):
        """When ratio=4, sqrt(4)=2, so factor=max(1,2)=2. Verifying sqrt is applied."""
        sp = SoakParams(
            base_durations={DataTier.PUBLIC: timedelta(minutes=10)},
            target_requests=400.0,
            observed_rate_rps=100.0,
            low_trust_threshold=0.5,
        )
        result = compute_soak_duration(DataTier.PUBLIC, 1.0, sp)
        # base=10min, trust_mult=1.0, ratio_factor=2.0
        expected = timedelta(minutes=10) * 1.0 * 2.0
        assert result == expected

    def test_goodhart_csd_ratio_nine_sqrt_three(self):
        """When ratio=9, sqrt(9)=3. Verifying the sqrt calculation is correct."""
        sp = SoakParams(
            base_durations={DataTier.PII: timedelta(minutes=10)},
            target_requests=90.0,
            observed_rate_rps=10.0,
            low_trust_threshold=0.5,
        )
        result = compute_soak_duration(DataTier.PII, 0.0, sp)
        # base=10min, trust_mult=2.0, ratio_factor=3.0
        expected = timedelta(minutes=10) * 2.0 * 3.0
        assert result == expected


# ── classify_node tests ─────────────────────────────────────────────────

class TestGoodhartClassifyNode:

    def test_goodhart_cn_pii_authorized_authoritative_high_trust(self):
        """Authoritative PII node with high trust should still be SOAK, not escalate."""
        nid = NodeId("N1")
        meta = make_node("N1", DataTier.PII, 0.9, [DataTier.PII], True)
        detail = make_detail("N1", DataTier.PII, 0.9, True, True, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.5)
        assert result == ActionCategory.SOAK

    def test_goodhart_cn_compliance_authorized_high_trust(self):
        """COMPLIANCE node must always be HUMAN_GATE regardless of authorization and trust."""
        meta = make_node("N1", DataTier.COMPLIANCE, 1.0, [DataTier.COMPLIANCE], False)
        detail = make_detail("N1", DataTier.COMPLIANCE, 1.0, False, True, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.5)
        assert result == ActionCategory.HUMAN_GATE

    def test_goodhart_cn_financial_authorized_high_trust(self):
        """FINANCIAL node must always be HUMAN_GATE even with full authorization and high trust."""
        meta = make_node("N1", DataTier.FINANCIAL, 1.0, [DataTier.FINANCIAL], False)
        detail = make_detail("N1", DataTier.FINANCIAL, 1.0, False, True, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.5)
        assert result == ActionCategory.HUMAN_GATE

    def test_goodhart_cn_public_low_trust_non_authoritative(self):
        """PUBLIC node with low trust but NOT authoritative should be AUTO_MERGE."""
        meta = make_node("N1", DataTier.PUBLIC, 0.1, [DataTier.PUBLIC], False)
        detail = make_detail("N1", DataTier.PUBLIC, 0.1, False, True, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.5)
        assert result == ActionCategory.AUTO_MERGE

    def test_goodhart_cn_pii_unauthorized_authoritative_low_trust(self):
        """Multiple escalation rules apply: result is still HUMAN_GATE."""
        meta = make_node("N1", DataTier.PII, 0.1, [DataTier.PUBLIC], True)
        detail = make_detail("N1", DataTier.PII, 0.1, True, False, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.5)
        assert result == ActionCategory.HUMAN_GATE

    def test_goodhart_cn_auth_unauthorized_low_trust_authoritative(self):
        """AUTH node must be HUMAN_GATE regardless of other conditions."""
        meta = make_node("N1", DataTier.AUTH, 0.1, [], True)
        detail = make_detail("N1", DataTier.AUTH, 0.1, True, False, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.9)
        assert result == ActionCategory.HUMAN_GATE

    def test_goodhart_cn_pii_authorized_non_authoritative_is_soak(self):
        """PII authorized non-authoritative yields SOAK with different trust values than visible tests."""
        meta = make_node("N1", DataTier.PII, 0.42, [DataTier.PII], False)
        detail = make_detail("N1", DataTier.PII, 0.42, False, True, ActionCategory.AUTO_MERGE, 0)
        result = classify_node(detail, meta, 0.3)
        assert result == ActionCategory.SOAK


# ── evaluate_blast tests ────────────────────────────────────────────────

class TestGoodhartEvaluateBlast:

    def test_goodhart_eb_soak_duration_positive_for_soak_action(self):
        """SOAK action must have a positive (not zero, not None) soak_duration."""
        nodes = [make_node("A", DataTier.PII, 0.8, [DataTier.PII], False)]
        g = make_graph(nodes)
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert result.action == ActionCategory.SOAK
        assert result.soak_duration is not None
        assert result.soak_duration > timedelta(0)

    def test_goodhart_eb_soak_matches_compute_soak_formula(self):
        """BlastResult.soak_duration must match compute_soak_duration formula."""
        nodes = [make_node("A", DataTier.PII, 0.7, [DataTier.PII], False)]
        g = make_graph(nodes)
        sp = default_soak_params(target_requests=100.0, observed_rate_rps=10.0)
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        # Manually compute expected soak
        base = sp.base_durations[DataTier.PII]
        trust_mult = 2.0 - 0.7
        ratio = sp.target_requests / sp.observed_rate_rps
        ratio_factor = max(1.0, math.sqrt(ratio))
        expected = base * trust_mult * ratio_factor
        assert result.soak_duration == expected

    def test_goodhart_eb_notifier_not_called_for_soak(self):
        """Notifier must NOT be called when action is SOAK."""
        nodes = [make_node("A", DataTier.PII, 0.8, [DataTier.PII], False)]
        g = make_graph(nodes)
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert result.action == ActionCategory.SOAK
        notifier.notify.assert_not_called()

    def test_goodhart_eb_notifier_receives_blast_result(self):
        """When HUMAN_GATE fires notifier, it must receive the actual BlastResult."""
        nodes = [make_node("A", DataTier.FINANCIAL, 0.8, [DataTier.FINANCIAL], False)]
        g = make_graph(nodes)
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert result.action == ActionCategory.HUMAN_GATE
        notifier.notify.assert_called_once()
        call_arg = notifier.notify.call_args[0][0]
        assert call_arg.origin_node == NodeId("A")

    def test_goodhart_eb_cycle_detected_propagated(self):
        """cycle_detected from BFS must be propagated to BlastResult."""
        nodes = [make_node("A"), make_node("B")]
        g = make_graph(nodes, [("A","B"),("B","A")])
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert result.cycle_detected is True

    def test_goodhart_eb_max_depth_reached_propagated(self):
        """max_depth_reached from traversal must be in BlastResult."""
        nodes = [make_node(c) for c in "ABCD"]
        g = make_graph(nodes, [("A","B"),("B","C"),("C","D")])
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=2)
        assert result.max_depth_reached == 2
        assert len(result.reachable_nodes) == 3

    def test_goodhart_eb_per_node_details_populated(self):
        """per_node_details must match reachable_nodes in count and node_ids."""
        nodes = [make_node("X"), make_node("Y"), make_node("Z")]
        g = make_graph(nodes, [("X","Y"),("Y","Z")])
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("X"), sp, notifier, max_depth=None)
        assert len(result.per_node_details) == len(result.reachable_nodes)
        detail_ids = {d.node_id for d in result.per_node_details}
        for nid in result.reachable_nodes:
            assert nid in detail_ids

    def test_goodhart_eb_contributing_nodes_in_blast_result(self):
        """BlastResult.contributing_nodes must be non-empty."""
        nodes = [make_node("A")]
        g = make_graph(nodes)
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert len(result.contributing_nodes) > 0

    def test_goodhart_eb_legal_flag_false_for_auto_merge(self):
        """AUTO_MERGE must have legal_flag=False and soak_duration=None."""
        nodes = [make_node("A", DataTier.PUBLIC, 0.9, [DataTier.PUBLIC], False)]
        g = make_graph(nodes)
        sp = default_soak_params()
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert result.action == ActionCategory.AUTO_MERGE
        assert result.legal_flag is False
        assert result.soak_duration is None

    def test_goodhart_eb_soak_duration_uses_highest_tier(self):
        """When action is SOAK, soak_duration must use highest_data_tier's base, not origin's."""
        a = make_node("A", DataTier.PUBLIC, 0.8, [DataTier.PUBLIC], False)
        b = make_node("B", DataTier.PII, 0.8, [DataTier.PII], False)
        g = make_graph([a, b], [("A","B")])
        sp = default_soak_params(target_requests=100.0, observed_rate_rps=100.0)
        notifier = MagicMock()
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        assert result.action == ActionCategory.SOAK
        # Should use PII base (highest tier), not PUBLIC base
        pii_base = sp.base_durations[DataTier.PII]
        public_base = sp.base_durations[DataTier.PUBLIC]
        # Trust score of origin? Contract says compute_soak_duration with tier=highest and trust_score
        # The soak duration should be at least using the PII base, not PUBLIC
        assert result.soak_duration is not None
        # If it used PUBLIC base, it would be much smaller
        assert result.soak_duration.total_seconds() > public_base.total_seconds()

    def test_goodhart_eb_computed_at_is_recent_utc(self):
        """computed_at must be a recent UTC datetime, not a hardcoded value."""
        nodes = [make_node("A")]
        g = make_graph(nodes)
        sp = default_soak_params()
        notifier = MagicMock()
        before = datetime.now(timezone.utc)
        result = evaluate_blast(g, NodeId("A"), sp, notifier, max_depth=None)
        after = datetime.now(timezone.utc)
        assert result.computed_at.tzinfo is not None
        assert before <= result.computed_at <= after


# ── add_node / add_edge tests ──────────────────────────────────────────

class TestGoodhartGraphMutation:

    def test_goodhart_add_node_preserves_existing_adjacency(self):
        """Replacing metadata via add_node must not clear existing edges."""
        a1 = make_node("A", DataTier.PUBLIC, 0.5)
        b = make_node("B")
        g = make_graph([a1, b], [("A","B")])
        # Replace A's metadata
        a2 = make_node("A", DataTier.PII, 0.9)
        add_node(g, a2)
        assert NodeId("B") in g.adjacency[NodeId("A")]
        assert g.metadata[NodeId("A")].data_tier == DataTier.PII
        assert g.metadata[NodeId("A")].trust_score == pytest.approx(0.9)

    def test_goodhart_add_edge_self_loop(self):
        """Self-loop edges should be allowed."""
        a = make_node("A")
        g = make_graph([a])
        add_edge(g, NodeId("A"), NodeId("A"))
        assert NodeId("A") in g.adjacency[NodeId("A")]

    def test_goodhart_add_node_empty_id_rejected(self):
        """NodeMetadata with empty node_id should be rejected."""
        with pytest.raises(Exception):
            make_node("")

    def test_goodhart_add_multiple_edges_different_targets(self):
        """Adding edges to different targets should all be present."""
        nodes = [make_node(c) for c in "ABCD"]
        g = make_graph(nodes, [("A","B"),("A","C"),("A","D")])
        adj = g.adjacency[NodeId("A")]
        assert NodeId("B") in adj
        assert NodeId("C") in adj
        assert NodeId("D") in adj
        assert len([x for x in adj if x == NodeId("B")]) == 1  # no duplication
