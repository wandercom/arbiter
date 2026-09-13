"""
Hidden adversarial acceptance tests for Configuration Loader component.
These tests catch implementations that 'teach to the test' by hardcoding
returns or skipping general validation logic.
"""
import os
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml

from arbiter.config import (
    load_config,
    get_config,
    reset_config,
    generate_default_config,
    validate_config_file,
    ArbiterConfig,
    ConfigurationError,
    ConfigNotLoadedError,
)


@pytest.fixture(autouse=True)
def clean_config_state(monkeypatch):
    """Reset config singleton and clean env vars before/after each test."""
    reset_config()
    # Remove any ARBITER_ env vars that might pollute tests
    for key in list(os.environ.keys()):
        if key.startswith("ARBITER_"):
            monkeypatch.delenv(key, raising=False)
    yield
    reset_config()


def _write_yaml(tmp_path, content):
    """Helper to write YAML content to a temp file and return its path."""
    p = tmp_path / "arbiter.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# --- Happy Path ---

class TestGoodhartHappyPath:

    def test_goodhart_env_override_nested_trust(self, tmp_path, monkeypatch):
        """Environment variables with nested delimiter __ should override deeply nested trust config fields, not just top-level fields."""
        monkeypatch.setenv("ARBITER_TRUST__DECAY_LAMBDA", "0.123")
        monkeypatch.chdir(tmp_path)
        cfg = load_config(None)
        assert cfg.trust.decay_lambda == pytest.approx(0.123)

    def test_goodhart_env_override_soak_target(self, tmp_path, monkeypatch):
        """Environment variable override should work for nested integer fields like soak.target_requests."""
        monkeypatch.setenv("ARBITER_SOAK__TARGET_REQUESTS", "42")
        monkeypatch.chdir(tmp_path)
        cfg = load_config(None)
        assert cfg.soak.target_requests == 42

    def test_goodhart_mid_range_valid_trust_values(self, tmp_path):
        """Non-boundary trust values should load successfully, verifying the validator is range-based not hardcoded for boundary values."""
        path = _write_yaml(tmp_path, """\
            trust:
              floor: 0.3
              authority_override_floor: 0.7
        """)
        cfg = load_config(path)
        assert cfg.trust.floor == pytest.approx(0.3)
        assert cfg.trust.authority_override_floor == pytest.approx(0.7)

    def test_goodhart_mid_range_port_valid(self, tmp_path):
        """Mid-range port values should be accepted, verifying port validation is range-based."""
        path = _write_yaml(tmp_path, """\
            api:
              port: 8080
            otlp:
              listen_port: 4317
              http_port: 4318
        """)
        cfg = load_config(path)
        assert cfg.api.port == 8080
        assert cfg.otlp.listen_port == 4317
        assert cfg.otlp.http_port == 4318

    def test_goodhart_full_yaml_override_all_fields(self, tmp_path):
        """A complete YAML file specifying every field with non-default values should have all values reflected."""
        path = _write_yaml(tmp_path, """\
            config_version: 1
            registry:
              path: /custom/registry.yaml
              append_only: false
            trust:
              floor: 0.2
              authority_override_floor: 0.8
              decay_lambda: 0.05
              conflict_trust_delta_threshold: 0.15
              taint_lock_tiers:
                - RESTRICTED
                - CRITICAL
            soak:
              base_durations:
                PUBLIC: 100
                INTERNAL: 200
                CONFIDENTIAL: 300
                RESTRICTED: 400
                CRITICAL: 500
              target_requests: 99
            otlp:
              listen_port: 5317
              http_port: 5318
            api:
              port: 9999
            classification_registry:
              path: /custom/classification.yaml
            human_gate:
              webhook_url: https://example.com/gate
              block_on_gate: false
            ledger:
              checksum_interval: 50
        """)
        cfg = load_config(path)
        assert cfg.trust.floor == pytest.approx(0.2)
        assert cfg.trust.authority_override_floor == pytest.approx(0.8)
        assert cfg.trust.decay_lambda == pytest.approx(0.05)
        assert cfg.trust.conflict_trust_delta_threshold == pytest.approx(0.15)
        assert cfg.soak.target_requests == 99
        assert cfg.api.port == 9999
        assert cfg.otlp.listen_port == 5317
        assert cfg.otlp.http_port == 5318
        assert cfg.ledger.checksum_interval == 50

    def test_goodhart_reset_then_load_fresh(self, tmp_path):
        """After reset_config() followed by load_config(), get_config() should return the newly loaded config."""
        path1 = tmp_path / "config1.yaml"
        path1.write_text("api:\n  port: 3000\n")
        path2 = tmp_path / "config2.yaml"
        path2.write_text("api:\n  port: 4000\n")

        load_config(str(path1))
        assert get_config().api.port == 3000
        reset_config()
        load_config(str(path2))
        assert get_config().api.port == 4000

    def test_goodhart_conflict_threshold_mid_value(self, tmp_path):
        """A mid-range conflict_trust_delta_threshold should be accepted."""
        path = _write_yaml(tmp_path, """\
            trust:
              conflict_trust_delta_threshold: 0.45
        """)
        cfg = load_config(path)
        assert cfg.trust.conflict_trust_delta_threshold == pytest.approx(0.45)


# --- Edge Cases ---

class TestGoodhartEdgeCases:

    def test_goodhart_load_nonexistent_explicit_path_falls_back(self, tmp_path):
        """When an explicit path is given but the file does not exist, the system should fall back to defaults."""
        nonexistent = str(tmp_path / "does_not_exist.yaml")
        cfg = load_config(nonexistent)
        assert cfg.config_version == 1
        assert isinstance(cfg, ArbiterConfig)

    def test_goodhart_single_taint_lock_tier(self, tmp_path):
        """A single valid DataClassificationTier in taint_lock_tiers should be accepted."""
        path = _write_yaml(tmp_path, """\
            trust:
              taint_lock_tiers:
                - CRITICAL
        """)
        cfg = load_config(path)
        assert len(cfg.trust.taint_lock_tiers) == 1
        assert str(cfg.trust.taint_lock_tiers[0]) == "CRITICAL" or cfg.trust.taint_lock_tiers[0].value == "CRITICAL" or cfg.trust.taint_lock_tiers[0] == "CRITICAL"

    def test_goodhart_authority_floor_slightly_below_one(self, tmp_path):
        """Non-boundary float equality for trust config values should work correctly."""
        path = _write_yaml(tmp_path, """\
            trust:
              floor: 0.999
              authority_override_floor: 0.999
        """)
        cfg = load_config(path)
        assert cfg.trust.floor == pytest.approx(0.999)
        assert cfg.trust.authority_override_floor == pytest.approx(0.999)

    def test_goodhart_port_two_valid(self, tmp_path):
        """Port value of 2 should be valid, not just hardcoded acceptance for port=1."""
        path = _write_yaml(tmp_path, """\
            api:
              port: 2
        """)
        cfg = load_config(path)
        assert cfg.api.port == 2

    def test_goodhart_port_65534_valid(self, tmp_path):
        """Port value of 65534 should be valid, not just hardcoded for 65535."""
        path = _write_yaml(tmp_path, """\
            otlp:
              listen_port: 65534
        """)
        cfg = load_config(path)
        assert cfg.otlp.listen_port == 65534

    def test_goodhart_soak_target_requests_large(self, tmp_path):
        """Large soak.target_requests values should be accepted without artificial upper bound."""
        path = _write_yaml(tmp_path, """\
            soak:
              target_requests: 1000000
        """)
        cfg = load_config(path)
        assert cfg.soak.target_requests == 1000000

    def test_goodhart_decay_lambda_large(self, tmp_path):
        """Large trust.decay_lambda values should be accepted since the only constraint is >= 0.0."""
        path = _write_yaml(tmp_path, """\
            trust:
              decay_lambda: 100.0
        """)
        cfg = load_config(path)
        assert cfg.trust.decay_lambda == pytest.approx(100.0)

    def test_goodhart_yaml_with_extra_keys(self, tmp_path):
        """YAML containing unknown/extra keys should either be ignored or raise a clean error, not crash."""
        path = _write_yaml(tmp_path, """\
            foo: bar
            baz: 123
        """)
        try:
            cfg = load_config(path)
            # If it loaded, it should still have valid defaults
            assert cfg.config_version == 1
        except (ConfigurationError, Exception):
            # A clean validation error is also acceptable
            pass

    def test_goodhart_trust_floor_epsilon_below_authority(self, tmp_path):
        """trust.floor very slightly below trust.authority_override_floor should be valid."""
        path = _write_yaml(tmp_path, """\
            trust:
              floor: 0.4999
              authority_override_floor: 0.5
        """)
        cfg = load_config(path)
        assert cfg.trust.floor == pytest.approx(0.4999)
        assert cfg.trust.authority_override_floor == pytest.approx(0.5)

    def test_goodhart_duplicate_taint_lock_tier(self, tmp_path):
        """Duplicate DataClassificationTier values in taint_lock_tiers should either be deduplicated or rejected."""
        path = _write_yaml(tmp_path, """\
            trust:
              taint_lock_tiers:
                - CRITICAL
                - CRITICAL
        """)
        try:
            cfg = load_config(path)
            # If accepted, duplicates may or may not be deduplicated
            # But the config should still be valid
            assert all(
                str(t) == "CRITICAL" or getattr(t, 'value', t) == "CRITICAL"
                for t in cfg.trust.taint_lock_tiers
            )
        except (ConfigurationError, Exception):
            # Rejecting duplicates is also acceptable
            pass


# --- Error Cases ---

class TestGoodhartErrorCases:

    def test_goodhart_soak_duration_negative_rejected(self, tmp_path):
        """Soak base duration values should reject negative integers."""
        path = _write_yaml(tmp_path, """\
            soak:
              base_durations:
                PUBLIC: -1
                INTERNAL: 200
                CONFIDENTIAL: 300
                RESTRICTED: 400
                CRITICAL: 500
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_soak_target_requests_negative(self, tmp_path):
        """Negative soak.target_requests should be rejected, not just zero."""
        path = _write_yaml(tmp_path, """\
            soak:
              target_requests: -5
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_ledger_checksum_interval_negative(self, tmp_path):
        """Negative ledger.checksum_interval should be rejected, not just zero."""
        path = _write_yaml(tmp_path, """\
            ledger:
              checksum_interval: -1
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_port_66000_rejected(self, tmp_path):
        """Port values just above 65535 should be rejected generically, not just 65536."""
        path = _write_yaml(tmp_path, """\
            api:
              port: 66000
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_trust_floor_slightly_above_authority(self, tmp_path):
        """Cross-field validation should catch trust.floor barely exceeding trust.authority_override_floor."""
        path = _write_yaml(tmp_path, """\
            trust:
              floor: 0.51
              authority_override_floor: 0.50
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_config_version_zero_rejected(self, tmp_path):
        """config_version=0 should be rejected, verifying the constraint is specifically for value 1."""
        path = _write_yaml(tmp_path, """\
            config_version: 0
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_config_version_negative_rejected(self, tmp_path):
        """config_version with a negative value should be rejected."""
        path = _write_yaml(tmp_path, """\
            config_version: -1
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_otlp_http_port_zero_rejected(self, tmp_path):
        """otlp.http_port of 0 should be rejected, verifying port validation applies to all port fields."""
        path = _write_yaml(tmp_path, """\
            otlp:
              http_port: 0
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)

    def test_goodhart_otlp_listen_port_above_65535_rejected(self, tmp_path):
        """otlp.listen_port above 65535 should be rejected, verifying upper bound on all port fields."""
        path = _write_yaml(tmp_path, """\
            otlp:
              listen_port: 70000
        """)
        with pytest.raises((ConfigurationError, Exception)):
            load_config(path)


# --- Invariants ---

class TestGoodhartInvariants:

    def test_goodhart_validate_does_not_set_singleton_when_none(self, tmp_path):
        """validate_config_file must not populate the singleton even when it was previously None."""
        reset_config()
        # Generate a valid config file
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        # Validate it - should NOT set the singleton
        validate_config_file(path)
        with pytest.raises(ConfigNotLoadedError):
            get_config()

    def test_goodhart_validate_does_not_overwrite_existing_singleton(self, tmp_path):
        """validate_config_file must not overwrite an existing singleton."""
        # Load defaults first
        default_path = str(tmp_path / "default.yaml")
        generate_default_config(default_path, overwrite=False)
        load_config(default_path)
        original = get_config()
        original_port = original.api.port

        # Create a different config file
        other_path = _write_yaml(tmp_path, """\
            api:
              port: 12345
        """)
        validated = validate_config_file(other_path)
        # Singleton should still be the original
        assert get_config() is original
        assert get_config().api.port == original_port

    def test_goodhart_generate_header_comment(self, tmp_path):
        """Generated config file must begin with a YAML comment line identifying it as auto-generated."""
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        content = Path(path).read_text()
        first_line = content.strip().split('\n')[0]
        assert first_line.startswith('#'), f"First line should be a comment, got: {first_line}"

    def test_goodhart_generate_contains_all_sections(self, tmp_path):
        """Generated YAML file must contain all top-level configuration sections."""
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        content = Path(path).read_text()
        data = yaml.safe_load(content)
        expected_keys = {'config_version', 'registry', 'trust', 'soak', 'otlp', 'api',
                         'classification_registry', 'human_gate', 'ledger'}
        assert expected_keys.issubset(set(data.keys())), \
            f"Missing keys: {expected_keys - set(data.keys())}"

    def test_goodhart_immutable_nested_config(self, tmp_path):
        """Nested config objects should also be frozen/immutable, not just the top-level."""
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        cfg = load_config(path)
        with pytest.raises(Exception):
            cfg.trust.floor = 0.99
        with pytest.raises(Exception):
            cfg.soak.target_requests = 999
        with pytest.raises(Exception):
            cfg.api.port = 1234

    def test_goodhart_load_config_returns_same_as_get_config(self, tmp_path):
        """The object returned by load_config() must be the exact same instance as get_config()."""
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        result = load_config(path)
        assert result is get_config()

    def test_goodhart_env_overrides_yaml_value(self, tmp_path, monkeypatch):
        """When both YAML and env var specify the same field, env var must take priority."""
        path = _write_yaml(tmp_path, """\
            api:
              port: 9090
        """)
        monkeypatch.setenv("ARBITER_API__PORT", "7070")
        cfg = load_config(path)
        assert cfg.api.port == 7070

    def test_goodhart_validate_returns_immutable(self, tmp_path):
        """The ArbiterConfig returned by validate_config_file should be frozen/immutable."""
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        cfg = validate_config_file(path)
        with pytest.raises(Exception):
            cfg.config_version = 99
        with pytest.raises(Exception):
            cfg.api.port = 1234

    def test_goodhart_soak_durations_all_tiers_in_yaml(self, tmp_path):
        """When YAML specifies custom soak base_durations, all five tier keys must be present."""
        path = _write_yaml(tmp_path, """\
            soak:
              base_durations:
                PUBLIC: 10
                INTERNAL: 20
                CONFIDENTIAL: 30
                RESTRICTED: 40
                CRITICAL: 50
              target_requests: 5
        """)
        cfg = load_config(path)
        durations = cfg.soak.base_durations
        # Check all five tiers are present (as attributes or dict keys)
        tier_names = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "CRITICAL"]
        if isinstance(durations, dict):
            for tier in tier_names:
                assert tier in durations or any(str(k) == tier for k in durations.keys())
        else:
            for tier in tier_names:
                assert hasattr(durations, tier) or hasattr(durations, tier.lower())

    def test_goodhart_generate_sort_keys_false(self, tmp_path):
        """Generated YAML must preserve definition order (sort_keys=False)."""
        path = str(tmp_path / "arbiter.yaml")
        generate_default_config(path, overwrite=False)
        content = Path(path).read_text()
        lines = content.split('\n')
        # Find top-level keys (lines that don't start with whitespace or #)
        top_keys = []
        for line in lines:
            if line and not line.startswith(' ') and not line.startswith('#') and ':' in line:
                key = line.split(':')[0].strip()
                top_keys.append(key)
        # config_version should come before registry, and registry before trust
        if 'config_version' in top_keys and 'registry' in top_keys:
            assert top_keys.index('config_version') < top_keys.index('registry'), \
                f"config_version should come before registry. Order: {top_keys}"
        if 'registry' in top_keys and 'trust' in top_keys:
            assert top_keys.index('registry') < top_keys.index('trust'), \
                f"registry should come before trust. Order: {top_keys}"
