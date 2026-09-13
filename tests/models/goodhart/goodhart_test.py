"""
Hidden adversarial acceptance tests for Shared Data Models & Enums.
These tests catch implementations that pass visible tests through shortcuts
(hardcoded returns, missing validation, etc.) rather than truly satisfying the contract.
"""

import json
import pytest
from pydantic import ValidationError

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
    CanaryRecord,
    ClassificationRule,
    TrustScoreRequest,
    BlastRadiusRequest,
    FindingsRequest,
    FeedbackReport,
    FeedbackReportSection,
    StigmerySignal,
    ConflictRecord,
    ErrorResponse,
    ValidationErrorDetail,
    Claim,
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


# ---- score_to_tier ----

def test_goodhart_score_to_tier_arbitrary_values_in_each_bracket():
    """score_to_tier must return the correct tier for arbitrary scores within each bracket,
    not just boundary values or midpoints that visible tests use."""
    assert score_to_tier(0.05) == TrustTier.PROBATIONARY
    assert score_to_tier(0.13) == TrustTier.PROBATIONARY
    assert score_to_tier(0.31) == TrustTier.LOW
    assert score_to_tier(0.25) == TrustTier.LOW
    assert score_to_tier(0.55) == TrustTier.ESTABLISHED
    assert score_to_tier(0.42) == TrustTier.ESTABLISHED
    assert score_to_tier(0.73) == TrustTier.HIGH
    assert score_to_tier(0.67) == TrustTier.HIGH
    assert score_to_tier(0.95) == TrustTier.TRUSTED
    assert score_to_tier(0.85) == TrustTier.TRUSTED


def test_goodhart_score_to_tier_float_precision_near_boundary():
    """score_to_tier must handle IEEE 754 floating point values extremely close to
    tier boundaries without misclassifying."""
    # 0.2 - epsilon should still be PROBATIONARY
    val = 0.19999999999999998
    if val < 0.2:
        assert score_to_tier(val) == TrustTier.PROBATIONARY
    # 0.2 + epsilon should be LOW
    val2 = 0.20000000000000001
    # In IEEE 754, this may round to 0.2 exactly
    assert score_to_tier(val2) == TrustTier.LOW

    # Near 0.8 boundary
    val3 = 0.7999999999999999
    if val3 < 0.8:
        assert score_to_tier(val3) == TrustTier.HIGH

    val4 = 0.8000000000000001
    if val4 >= 0.8:
        assert score_to_tier(val4) == TrustTier.TRUSTED


# ---- create_trust_ledger_entry ----

def test_goodhart_create_entry_score_after_exact_clamping():
    """score_after must be exactly clamp(score_before + weight, 0, 1) for various
    non-trivial combinations."""
    e1 = create_trust_ledger_entry('nodeA', TrustEventType.AUDIT_PASS, 0.15, 0.3, 5, 'test')
    assert e1.score_after == pytest.approx(0.45, abs=1e-15)

    e2 = create_trust_ledger_entry('nodeB', TrustEventType.DECAY, -0.3, 0.5, 6, 'decay')
    assert e2.score_after == pytest.approx(0.2, abs=1e-15)

    e3 = create_trust_ledger_entry('nodeC', TrustEventType.INITIAL, 0.0, 0.7, 7, 'init')
    assert e3.score_after == pytest.approx(0.7, abs=1e-15)


def test_goodhart_create_entry_score_before_preserved():
    """The returned entry must preserve the exact score_before value provided."""
    val = 0.3333333333333333
    e1 = create_trust_ledger_entry('n1', TrustEventType.AUDIT_PASS, 0.1, val, 0, 'test')
    assert e1.score_before == val

    # 0.1 + 0.2 in IEEE 754 is not exactly 0.3
    val2 = 0.1 + 0.2
    e2 = create_trust_ledger_entry('n2', TrustEventType.AUDIT_PASS, 0.0, val2, 1, 'test')
    assert e2.score_before == val2


def test_goodhart_create_entry_node_field_matches_input():
    """The returned entry's node field must exactly match the input node ID string."""
    e1 = create_trust_ledger_entry('a', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    assert e1.node == 'a'

    e2 = create_trust_ledger_entry('A.B-C_123', TrustEventType.AUDIT_PASS, 0.1, 0.5, 1, 'test')
    assert e2.node == 'A.B-C_123'

    long_node = 'x' * 255
    e3 = create_trust_ledger_entry(long_node, TrustEventType.AUDIT_PASS, 0.1, 0.5, 2, 'test')
    assert e3.node == long_node


def test_goodhart_create_entry_node_id_max_length_exceeded():
    """NodeId must reject strings longer than 255 characters."""
    long_node = 'x' * 256
    with pytest.raises((ValueError, ValidationError)):
        create_trust_ledger_entry(long_node, TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')


def test_goodhart_create_entry_node_id_special_chars():
    """NodeId must reject strings containing characters outside [a-zA-Z0-9._-]."""
    for invalid_char in ['/', '@', ':', '!', ' ', '#', '$', '(', ')']:
        with pytest.raises((ValueError, ValidationError)):
            create_trust_ledger_entry(
                f'node{invalid_char}bad', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test'
            )


def test_goodhart_create_entry_event_field_matches_input():
    """The returned entry's event field must exactly match the input TrustEventType."""
    e1 = create_trust_ledger_entry('n', TrustEventType.TAINT_DETECTED, 0.0, 0.5, 0, 'test')
    assert e1.event == TrustEventType.TAINT_DETECTED

    e2 = create_trust_ledger_entry('n', TrustEventType.CANARY_TRIGGERED, 0.0, 0.5, 1, 'test')
    assert e2.event == TrustEventType.CANARY_TRIGGERED

    e3 = create_trust_ledger_entry('n', TrustEventType.MANUAL_OVERRIDE, 0.0, 0.5, 2, 'test')
    assert e3.event == TrustEventType.MANUAL_OVERRIDE


def test_goodhart_create_entry_detail_preserved():
    """The returned entry's detail field must exactly preserve the input detail string."""
    e1 = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'hello\nworld')
    assert e1.detail == 'hello\nworld'

    e2 = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, 0.1, 0.5, 1, 'café ñ')
    assert e2.detail == 'café ñ'

    e3 = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, 0.1, 0.5, 2, '{"json": true}')
    assert e3.detail == '{"json": true}'


def test_goodhart_create_entry_sequence_zero_valid():
    """Sequence number 0 is a valid starting value and must be accepted and preserved."""
    e = create_trust_ledger_entry('n', TrustEventType.INITIAL, 0.5, 0.0, 0, 'init')
    assert e.sequence_number == 0


def test_goodhart_create_entry_sequence_large_value():
    """Large sequence numbers must be accepted and preserved exactly."""
    e = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, 0.1, 0.5, 999999999, 'test')
    assert e.sequence_number == 999999999


def test_goodhart_create_entry_other_events_allow_empty_detail():
    """Events other than AUDIT_FAIL and ACCESS_VIOLATION must accept empty detail strings."""
    for event_type in [
        TrustEventType.AUDIT_PASS,
        TrustEventType.DECAY,
        TrustEventType.INITIAL,
        TrustEventType.CONSISTENCY_CHECK,
        TrustEventType.TAINT_DETECTED,
        TrustEventType.CANARY_TRIGGERED,
        TrustEventType.MANUAL_OVERRIDE,
    ]:
        e = create_trust_ledger_entry('n', event_type, 0.0, 0.5, 0, '')
        assert e.detail == ''


def test_goodhart_create_entry_weight_zero():
    """A weight of exactly 0.0 should produce score_after equal to score_before."""
    e = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, 0.0, 0.5, 0, 'test')
    assert e.score_after == 0.5


def test_goodhart_create_entry_negative_weight_exact_clamp():
    """score_before=0.1 with weight=-0.5 should clamp score_after to 0.0."""
    e = create_trust_ledger_entry('n', TrustEventType.DECAY, -0.5, 0.1, 0, 'decay')
    assert e.score_after == 0.0


# ---- create_ledger_checkpoint ----

def test_goodhart_checkpoint_checksum_preserved():
    """The checkpoint must preserve the exact checksum string provided."""
    c1 = create_ledger_checkpoint(0, 'a' * 64, 1)
    assert c1.checksum == 'a' * 64

    c2 = create_ledger_checkpoint(1, '0123456789abcdef' * 4, 5)
    assert c2.checksum == '0123456789abcdef' * 4


def test_goodhart_checkpoint_entry_count_preserved():
    """The checkpoint must preserve the exact entry_count value."""
    c1 = create_ledger_checkpoint(0, 'a' * 64, 1)
    assert c1.entry_count == 1

    c2 = create_ledger_checkpoint(0, 'a' * 64, 1000000)
    assert c2.entry_count == 1000000


def test_goodhart_checkpoint_negative_entry_count():
    """Negative entry_count must be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        create_ledger_checkpoint(0, 'a' * 64, -1)


def test_goodhart_checkpoint_checksum_with_invalid_hex_chars():
    """Checksums containing non-hex characters must be rejected even if 64 chars long."""
    with pytest.raises((ValueError, ValidationError)):
        create_ledger_checkpoint(0, 'g' * 64, 1)


def test_goodhart_checkpoint_checksum_65_chars():
    """Checksums that are 65 characters (too long by one) must be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        create_ledger_checkpoint(0, 'a' * 65, 1)


def test_goodhart_checkpoint_sequence_preserved():
    """The checkpoint must preserve the exact sequence_number."""
    c = create_ledger_checkpoint(42, 'b' * 64, 10)
    assert c.sequence_number == 42


# ---- build_access_graph ----

def test_goodhart_build_graph_node_ids_match_keys():
    """In the returned AccessGraph, every node's id field must match its dict key."""
    nodes = {}
    for name in ['alpha', 'beta', 'gamma']:
        nodes[name] = AccessGraphNode(
            id=name, data_access=[], authority_domains=[], edges=[],
            trust_tier=TrustTier.LOW, metadata={}
        )
    g = build_access_graph(nodes)
    for key, node in g.nodes.items():
        assert node.id == key


def test_goodhart_build_graph_mutual_edges():
    """Graph with bidirectional edges between nodes must be valid."""
    nodes = {
        'a': AccessGraphNode(id='a', data_access=[], authority_domains=[], edges=['b'],
                             trust_tier=TrustTier.LOW, metadata={}),
        'b': AccessGraphNode(id='b', data_access=[], authority_domains=[], edges=['a', 'c'],
                             trust_tier=TrustTier.LOW, metadata={}),
        'c': AccessGraphNode(id='c', data_access=[], authority_domains=[], edges=['b'],
                             trust_tier=TrustTier.LOW, metadata={}),
    }
    g = build_access_graph(nodes)
    assert len(g.nodes) == 3


def test_goodhart_build_graph_fully_connected():
    """A fully connected graph where every node has edges to all other nodes must be valid."""
    names = ['n1', 'n2', 'n3', 'n4']
    nodes = {}
    for name in names:
        others = [n for n in names if n != name]
        nodes[name] = AccessGraphNode(
            id=name, data_access=[], authority_domains=[], edges=others,
            trust_tier=TrustTier.ESTABLISHED, metadata={}
        )
    g = build_access_graph(nodes)
    assert len(g.nodes) == 4
    for key, node in g.nodes.items():
        for edge in node.edges:
            assert edge in g.nodes


def test_goodhart_access_graph_mutable():
    """AccessGraph must be mutable (not frozen) to allow incremental construction."""
    nodes = {
        'x': AccessGraphNode(id='x', data_access=[], authority_domains=[], edges=[],
                             trust_tier=TrustTier.LOW, metadata={})
    }
    g = build_access_graph(nodes)
    # Should be able to reassign version
    g.version = "2.0"
    assert g.version == "2.0"


def test_goodhart_build_graph_version_field():
    """The returned AccessGraph must have a version field that is a string."""
    nodes = {
        'x': AccessGraphNode(id='x', data_access=[], authority_domains=[], edges=[],
                             trust_tier=TrustTier.LOW, metadata={})
    }
    g = build_access_graph(nodes)
    assert isinstance(g.version, str)


# ---- parse_ledger_line ----

def test_goodhart_parse_ledger_line_extra_fields_rejected():
    """JSON with all required fields plus extra unknown fields must be rejected."""
    entry = create_trust_ledger_entry('node1', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    serialized = serialize_ledger_line(entry)
    data = json.loads(serialized)
    data['extra_unexpected_field'] = 'should_fail'
    line_with_extra = json.dumps(data)
    with pytest.raises((ValueError, ValidationError, Exception)):
        parse_ledger_line(line_with_extra)


def test_goodhart_parse_ledger_line_missing_required_field():
    """JSON missing a required field for TrustLedgerEntry must be rejected."""
    entry = create_trust_ledger_entry('node1', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    serialized = serialize_ledger_line(entry)
    data = json.loads(serialized)
    # Remove a required field
    if 'weight' in data:
        del data['weight']
    elif 'node' in data:
        del data['node']
    line_missing = json.dumps(data)
    with pytest.raises((ValueError, ValidationError, Exception)):
        parse_ledger_line(line_missing)


def test_goodhart_parse_ledger_line_invalid_event_type():
    """JSON with all TrustLedgerEntry fields but an invalid event type string must be rejected."""
    entry = create_trust_ledger_entry('node1', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    serialized = serialize_ledger_line(entry)
    data = json.loads(serialized)
    data['event'] = 'NONEXISTENT_EVENT'
    line_bad_event = json.dumps(data)
    with pytest.raises((ValueError, ValidationError, Exception)):
        parse_ledger_line(line_bad_event)


def test_goodhart_parse_line_tab_only_whitespace():
    """A line containing only tabs should be treated as empty/whitespace-only."""
    with pytest.raises((ValueError, ValidationError, Exception)):
        parse_ledger_line('\t\t')


def test_goodhart_parse_line_json_array():
    """Valid JSON that is an array (not object) must be rejected."""
    with pytest.raises((ValueError, ValidationError, Exception)):
        parse_ledger_line('[1, 2, 3]')


def test_goodhart_parse_line_json_null():
    """The JSON literal 'null' must be rejected."""
    with pytest.raises((ValueError, ValidationError, Exception)):
        parse_ledger_line('null')


# ---- serialize_ledger_line ----

def test_goodhart_serialize_roundtrip_checkpoint():
    """Serialization and parsing of LedgerCheckpoint must round-trip exactly."""
    cp = create_ledger_checkpoint(10, 'ab' * 32, 5)
    serialized = serialize_ledger_line(cp)
    parsed = parse_ledger_line(serialized)
    assert isinstance(parsed, LedgerCheckpoint)
    assert parsed.sequence_number == cp.sequence_number
    assert parsed.checksum == cp.checksum
    assert parsed.entry_count == cp.entry_count
    assert parsed.ts == cp.ts


def test_goodhart_serialize_deterministic_repeated_calls():
    """Calling serialize_ledger_line multiple times on the same entry must produce identical output."""
    entry = create_trust_ledger_entry('svc', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    s1 = serialize_ledger_line(entry)
    s2 = serialize_ledger_line(entry)
    s3 = serialize_ledger_line(entry)
    assert s1 == s2
    assert s2 == s3


def test_goodhart_serialize_entry_is_valid_json():
    """Serialized output must be parseable as valid JSON by json.loads."""
    entry = create_trust_ledger_entry('svc', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    cp = create_ledger_checkpoint(0, 'c' * 64, 1)

    data1 = json.loads(serialize_ledger_line(entry))
    assert isinstance(data1, dict)

    data2 = json.loads(serialize_ledger_line(cp))
    assert isinstance(data2, dict)


# ---- create_error_response ----

def test_goodhart_error_response_only_node_provided():
    """When only node is provided, details should contain 'node' but not 'field' or 'domain'
    (or those keys should be absent/empty)."""
    resp = create_error_response('err1', 'something failed', 'svc-1', '', '')
    assert 'node' in resp.details
    assert resp.details['node'] == 'svc-1'
    # Empty-string context fields should NOT appear in details
    # (or if they do, they should be empty — the key point is node IS there)


def test_goodhart_error_response_only_field_provided():
    """When only field is provided, details should contain 'field' key with the given value."""
    resp = create_error_response('err2', 'bad field', '', 'email', '')
    assert 'field' in resp.details
    assert resp.details['field'] == 'email'


def test_goodhart_error_response_only_domain_provided():
    """When only domain is provided, details should contain 'domain' key with the given value."""
    resp = create_error_response('err3', 'bad domain', '', '', 'auth/core')
    assert 'domain' in resp.details
    assert resp.details['domain'] == 'auth/core'


def test_goodhart_error_response_preserves_error_code():
    """The returned ErrorResponse must preserve the exact error_code string."""
    resp = create_error_response('CUSTOM_CODE_42', 'msg', 'n', '', '')
    assert resp.error_code == 'CUSTOM_CODE_42'


def test_goodhart_error_response_preserves_message():
    """The returned ErrorResponse must preserve the exact message string."""
    msg = 'A long detailed error message with special chars: <>&'
    resp = create_error_response('err', msg, 'n', '', '')
    assert resp.message == msg


# ---- validate_canary_fingerprint ----

def test_goodhart_canary_uuid_v4_variant_check():
    """UUID v4 validation must check the variant bits (position 19 must be 8, 9, a, or b)."""
    # Valid version nibble (4) but invalid variant (c instead of 8/9/a/b)
    bad_variant = '12345678-1234-4abc-cabc-123456789abc'
    with pytest.raises((ValueError, Exception)):
        validate_canary_fingerprint(bad_variant)


def test_goodhart_canary_uuid_embedded_in_complex_string():
    """UUID v4 embedded within a longer complex string must still be detected as valid."""
    fp = 'prefix-data-12345678-1234-4abc-9abc-123456789abc-suffix-more'
    assert validate_canary_fingerprint(fp) is True


def test_goodhart_canary_uuid_uppercase_rejected():
    """UUID v4 with uppercase hex characters should not match the lowercase-only contract regex."""
    upper_uuid = '12345678-1234-4ABC-9ABC-123456789ABC'
    with pytest.raises((ValueError, Exception)):
        validate_canary_fingerprint(upper_uuid)


def test_goodhart_canary_uuid_v5_rejected():
    """UUID v5 (version nibble = 5) must be rejected as not v4."""
    v5_uuid = '12345678-1234-5abc-9abc-123456789abc'
    with pytest.raises((ValueError, Exception)):
        validate_canary_fingerprint(v5_uuid)


def test_goodhart_canary_uuid_v3_rejected():
    """UUID v3 (version nibble = 3) must be rejected as not v4."""
    v3_uuid = '12345678-1234-3abc-9abc-123456789abc'
    with pytest.raises((ValueError, Exception)):
        validate_canary_fingerprint(v3_uuid)


# ---- classify_field ----

def test_goodhart_classify_field_fnmatch_wildcard_various():
    """fnmatch rules must support standard wildcard patterns like '*_ssn' matching any prefix."""
    rule = ClassificationRule(
        field_pattern='*_ssn', data_tier=DataTier.PII, is_regex=False, description='SSN fields'
    )
    assert classify_field('user_ssn', [rule]) == DataTier.PII
    assert classify_field('employee_ssn', [rule]) == DataTier.PII
    assert classify_field('ssn_number', [rule]) == DataTier.PUBLIC  # doesn't match *_ssn


def test_goodhart_classify_field_fnmatch_question_mark():
    """fnmatch rules must support '?' single-character wildcard."""
    rule = ClassificationRule(
        field_pattern='field_?', data_tier=DataTier.FINANCIAL, is_regex=False, description='single char'
    )
    assert classify_field('field_a', [rule]) == DataTier.FINANCIAL
    assert classify_field('field_ab', [rule]) == DataTier.PUBLIC  # '?' matches exactly one char


def test_goodhart_classify_field_third_rule_matches():
    """When the first two rules don't match but the third does, the third rule's tier must be returned."""
    rules = [
        ClassificationRule(field_pattern='aaa*', data_tier=DataTier.AUTH, is_regex=False, description='r1'),
        ClassificationRule(field_pattern='bbb*', data_tier=DataTier.FINANCIAL, is_regex=False, description='r2'),
        ClassificationRule(field_pattern='ccc*', data_tier=DataTier.COMPLIANCE, is_regex=False, description='r3'),
    ]
    assert classify_field('ccc_field', rules) == DataTier.COMPLIANCE


def test_goodhart_classify_field_regex_no_match_returns_public():
    """When regex rules exist but none match, PUBLIC must be returned."""
    rule = ClassificationRule(
        field_pattern='^zzz_.*$', data_tier=DataTier.AUTH, is_regex=True, description='no match'
    )
    assert classify_field('user_email', [rule]) == DataTier.PUBLIC


def test_goodhart_classify_field_regex_matches():
    """Regex rule that matches must return the correct tier."""
    rule = ClassificationRule(
        field_pattern='.*password.*', data_tier=DataTier.AUTH, is_regex=True, description='password fields'
    )
    assert classify_field('user_password_hash', [rule]) == DataTier.AUTH


def test_goodhart_classify_field_order_dependent_mixed_types():
    """First-match-wins must work correctly when mixing fnmatch and regex rules."""
    rules = [
        ClassificationRule(field_pattern='secret_*', data_tier=DataTier.AUTH, is_regex=False, description='fnmatch'),
        ClassificationRule(field_pattern='.*secret.*', data_tier=DataTier.COMPLIANCE, is_regex=True, description='regex'),
    ]
    # 'secret_key' matches first rule (fnmatch), should return AUTH not COMPLIANCE
    assert classify_field('secret_key', rules) == DataTier.AUTH


# ---- Frozen/extra='forbid' invariants for structs NOT covered by visible tests ----

def test_goodhart_consistency_finding_frozen_and_forbid():
    """ConsistencyFinding must be frozen (no mutation) and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    cf = ConsistencyFinding(
        ts=ts, node='svc1', severity=FindingSeverity.HIGH,
        field='name', adapter_value='real', claimed_value='fake', detail='mismatch'
    )
    with pytest.raises(Exception):
        cf.node = 'other'

    with pytest.raises((ValidationError, Exception)):
        ConsistencyFinding(
            ts=ts, node='svc1', severity=FindingSeverity.HIGH,
            field='name', adapter_value='real', claimed_value='fake', detail='mismatch',
            extra_field='nope'
        )


def test_goodhart_access_finding_frozen_and_forbid():
    """AccessFinding must be frozen and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    af = AccessFinding(
        ts=ts, node='svc1', severity=FindingSeverity.MEDIUM,
        data_tier=DataTier.PII, authority_domain='auth/domain',
        violation_type='unauthorized', detail='bad access'
    )
    with pytest.raises(Exception):
        af.node = 'other'

    with pytest.raises((ValidationError, Exception)):
        AccessFinding(
            ts=ts, node='svc1', severity=FindingSeverity.MEDIUM,
            data_tier=DataTier.PII, authority_domain='auth/domain',
            violation_type='unauthorized', detail='bad access',
            bonus='nope'
        )


def test_goodhart_taint_finding_frozen_and_forbid():
    """TaintFinding must be frozen and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    tf = TaintFinding(
        ts=ts, source_node='src', sink_node='sink',
        severity=FindingSeverity.CRITICAL, data_tier=DataTier.FINANCIAL,
        path=['src', 'mid', 'sink'], detail='data leak'
    )
    with pytest.raises(Exception):
        tf.source_node = 'other'

    with pytest.raises((ValidationError, Exception)):
        TaintFinding(
            ts=ts, source_node='src', sink_node='sink',
            severity=FindingSeverity.CRITICAL, data_tier=DataTier.FINANCIAL,
            path=['src', 'mid', 'sink'], detail='data leak',
            extra='nope'
        )


def test_goodhart_canary_record_frozen_and_forbid():
    """CanaryRecord must be frozen and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    cr = CanaryRecord(
        ts=ts, canary_id='c1', fingerprint='canary-12345678-1234-4abc-8abc-123456789abc',
        data_tier=DataTier.PII, target_node='node1',
        triggered=False, triggered_at='', triggered_by_node=''
    )
    with pytest.raises(Exception):
        cr.triggered = True

    with pytest.raises((ValidationError, Exception)):
        CanaryRecord(
            ts=ts, canary_id='c1', fingerprint='canary-12345678-1234-4abc-8abc-123456789abc',
            data_tier=DataTier.PII, target_node='node1',
            triggered=False, triggered_at='', triggered_by_node='',
            extra_field='nope'
        )


def test_goodhart_classification_rule_frozen_and_forbid():
    """ClassificationRule must be frozen and forbid extra fields."""
    rule = ClassificationRule(
        field_pattern='*', data_tier=DataTier.PUBLIC, is_regex=False, description='all'
    )
    with pytest.raises(Exception):
        rule.field_pattern = 'changed'

    with pytest.raises((ValidationError, Exception)):
        ClassificationRule(
            field_pattern='*', data_tier=DataTier.PUBLIC, is_regex=False,
            description='all', extra='nope'
        )


def test_goodhart_trust_score_request_extra_forbid():
    """TrustScoreRequest must reject extra fields during construction."""
    with pytest.raises((ValidationError, Exception)):
        TrustScoreRequest(node='svc1', extra_field='nope')


def test_goodhart_blast_radius_request_extra_forbid():
    """BlastRadiusRequest must reject extra fields during construction."""
    with pytest.raises((ValidationError, Exception)):
        BlastRadiusRequest(node='svc1', max_depth=3, extra_field='nope')


def test_goodhart_findings_request_extra_forbid():
    """FindingsRequest must reject extra fields during construction."""
    with pytest.raises((ValidationError, Exception)):
        FindingsRequest(node='svc1', severity_min=FindingSeverity.LOW, limit=10, extra_field='nope')


def test_goodhart_conflict_record_frozen_and_forbid():
    """ConflictRecord must be frozen and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    cr = ConflictRecord(
        ts=ts, conflict_id='c1', nodes=['a', 'b'],
        authority_domain='dom/1', conflict_type='overlap',
        detail='conflict', resolved=False
    )
    with pytest.raises(Exception):
        cr.resolved = True

    with pytest.raises((ValidationError, Exception)):
        ConflictRecord(
            ts=ts, conflict_id='c1', nodes=['a', 'b'],
            authority_domain='dom/1', conflict_type='overlap',
            detail='conflict', resolved=False, extra='nope'
        )


def test_goodhart_stigmery_signal_frozen_and_forbid():
    """StigmerySignal must be frozen and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    ss = StigmerySignal(
        ts=ts, signal_id='s1', source_node='node1',
        signal_type='alert', payload={}, ttl_seconds=300
    )
    with pytest.raises(Exception):
        ss.ttl_seconds = 600

    with pytest.raises((ValidationError, Exception)):
        StigmerySignal(
            ts=ts, signal_id='s1', source_node='node1',
            signal_type='alert', payload={}, ttl_seconds=300, extra='nope'
        )


def test_goodhart_feedback_report_frozen_and_forbid():
    """FeedbackReport must be frozen and forbid extra fields."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    fr = FeedbackReport(
        ts=ts, report_id='r1', sections=[], total_findings=0, generated_by='node1'
    )
    with pytest.raises(Exception):
        fr.total_findings = 5

    with pytest.raises((ValidationError, Exception)):
        FeedbackReport(
            ts=ts, report_id='r1', sections=[], total_findings=0,
            generated_by='node1', extra='nope'
        )


def test_goodhart_feedback_report_section_frozen_and_forbid():
    """FeedbackReportSection must be frozen and forbid extra fields."""
    frs = FeedbackReportSection(
        section_name='summary', content='all good', findings_count=0, metadata={}
    )
    with pytest.raises(Exception):
        frs.content = 'changed'

    with pytest.raises((ValidationError, Exception)):
        FeedbackReportSection(
            section_name='summary', content='all good',
            findings_count=0, metadata={}, extra='nope'
        )


# ---- Enum string values stability ----

def test_goodhart_enum_values_are_strings():
    """All enum members must be instances of str (StrEnum)."""
    assert isinstance(TrustTier.PROBATIONARY, str)
    assert isinstance(TrustTier.TRUSTED, str)
    assert isinstance(DataTier.PII, str)
    assert isinstance(DataTier.COMPLIANCE, str)
    assert isinstance(BlastTier.SOAK, str)
    assert isinstance(BlastTier.HUMAN_GATE, str)
    assert isinstance(FindingSeverity.CRITICAL, str)
    assert isinstance(FindingSeverity.INFO, str)
    assert isinstance(TrustEventType.DECAY, str)
    assert isinstance(TrustEventType.INITIAL, str)


def test_goodhart_enum_json_serialization():
    """Enums used in Pydantic models should serialize to their string values in JSON."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    node = AccessGraphNode(
        id='test', data_access=[], authority_domains=[], edges=[],
        trust_tier=TrustTier.HIGH, metadata={}
    )
    dumped = node.model_dump()
    # trust_tier should be serializable as string
    tier_val = dumped['trust_tier']
    assert isinstance(tier_val, str) or hasattr(tier_val, 'value')


def test_goodhart_enum_member_count_exact():
    """Each enum must have exactly the specified number of members — no extras."""
    assert len(TrustTier) == 5
    assert len(DataTier) == 5
    assert len(BlastTier) == 3
    assert len(FindingSeverity) == 5
    assert len(TrustEventType) == 9


# ---- IEEE 754 fidelity across the full pipeline ----

def test_goodhart_ieee754_subnormal_roundtrip():
    """Very small float values near zero must round-trip through create/serialize/parse."""
    tiny = 1e-15
    entry = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, tiny, 0.0, 0, 'tiny')
    assert entry.score_after == tiny
    serialized = serialize_ledger_line(entry)
    parsed = parse_ledger_line(serialized)
    assert parsed.score_after == tiny
    assert parsed.score_before == 0.0


def test_goodhart_ieee754_one_third_roundtrip():
    """The value 1/3 (repeating decimal) must round-trip with full IEEE 754 fidelity."""
    val = 1.0 / 3.0
    entry = create_trust_ledger_entry('n', TrustEventType.AUDIT_PASS, 0.0, val, 0, 'third')
    serialized = serialize_ledger_line(entry)
    parsed = parse_ledger_line(serialized)
    assert parsed.score_before == val


# ---- Misc edge cases ----

def test_goodhart_node_id_underscore_only():
    """A NodeId consisting of just underscores should be valid per [a-zA-Z0-9._-]+."""
    e = create_trust_ledger_entry('___', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    assert e.node == '___'


def test_goodhart_node_id_single_char():
    """A single character NodeId should be valid."""
    e = create_trust_ledger_entry('Z', TrustEventType.AUDIT_PASS, 0.1, 0.5, 0, 'test')
    assert e.node == 'Z'


def test_goodhart_score_to_tier_at_0999():
    """score_to_tier at 0.999 should return TRUSTED (not out of range)."""
    assert score_to_tier(0.999) == TrustTier.TRUSTED


def test_goodhart_score_to_tier_at_001():
    """score_to_tier at 0.01 should return PROBATIONARY."""
    assert score_to_tier(0.01) == TrustTier.PROBATIONARY


def test_goodhart_build_graph_invalid_node_id_key():
    """A key in the dict that is not a valid NodeId must be rejected."""
    nodes = {
        'valid-node': AccessGraphNode(
            id='valid-node', data_access=[], authority_domains=[], edges=[],
            trust_tier=TrustTier.LOW, metadata={}
        ),
        'invalid node': AccessGraphNode(
            id='valid-other', data_access=[], authority_domains=[], edges=[],
            trust_tier=TrustTier.LOW, metadata={}
        ),
    }
    with pytest.raises((ValueError, ValidationError, Exception)):
        build_access_graph(nodes)
