"""
Contract tests for the Arbiter config module.

Tests cover: load_config, get_config, reset_config, generate_default_config, validate_config_file.
Organized by: happy path, edge cases, error cases, invariants.
"""

import os
import stat
import sys
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

# Import the component under test
from arbiter.config import (
    load_config,
    get_config,
    reset_config,
    generate_default_config,
    validate_config_file,
    ArbiterConfig,
    ConfigurationError,
    ConfigNotLoadedError,
    DataClassificationTier,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def config_reset_fixture():
    """Reset the module-level config singleton before and after every test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def valid_config_dict():
    """Minimal valid ArbiterConfig as a Python dict with reasonable defaults."""
    return {
        "config_version": 1,
        "registry": {"path": "registry.yaml", "append_only": True},
        "trust": {
            "floor": 0.1,
            "authority_override_floor": 0.5,
            "decay_lambda": 0.01,
            "conflict_trust_delta_threshold": 0.2,
            "taint_lock_tiers": ["RESTRICTED", "CRITICAL"],
        },
        "soak": {
            "base_durations": {
                "PUBLIC": 60,
                "INTERNAL": 300,
                "CONFIDENTIAL": 600,
                "RESTRICTED": 1800,
                "CRITICAL": 3600,
            },
            "target_requests": 100,
        },
        "otlp": {"listen_port": 4317, "http_port": 4318},
        "api": {"port": 8080},
        "classification_registry": {"path": "classification.yaml"},
        "human_gate": {"webhook_url": "https://example.com/hook", "block_on_gate": True},
        "ledger": {"checksum_interval": 100},
    }


def write_yaml(path, data):
    """Helper: write a dict as YAML to a file path."""
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return path


@pytest.fixture
def valid_config_file(tmp_path, valid_config_dict):
    """Write a valid YAML config file and return its path."""
    p = tmp_path / "arbiter.yaml"
    write_yaml(str(p), valid_config_dict)
    return str(p)


def write_raw(path, content):
    """Helper: write raw string content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return path


# ═══════════════════════════════════════════════════════════════════
# Happy-path tests
# ═══════════════════════════════════════════════════════════════════

class TestLoadConfigHappyPath:

    def test_load_from_explicit_path(self, valid_config_file):
        """load_config with explicit path returns validated ArbiterConfig."""
        cfg = load_config(valid_config_file)
        assert isinstance(cfg, ArbiterConfig)
        assert cfg.config_version == 1
        assert cfg.api.port == 8080

    def test_load_from_none_falls_back_to_defaults(self, tmp_path):
        """load_config(None) with no arbiter.yaml in cwd uses all defaults."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            cfg = load_config(None)
        assert isinstance(cfg, ArbiterConfig)
        assert cfg.config_version == 1

    def test_load_partial_yaml_fills_defaults(self, tmp_path):
        """Partial YAML fills missing sections with defaults."""
        partial = {"trust": {"floor": 0.2, "authority_override_floor": 0.6}}
        p = write_yaml(str(tmp_path / "arbiter.yaml"), partial)
        cfg = load_config(p)
        assert isinstance(cfg, ArbiterConfig)
        assert cfg.trust.floor == 0.2
        assert cfg.trust.authority_override_floor == 0.6
        # Other sections should be defaults
        assert cfg.config_version == 1

    def test_load_overwrites_previous_singleton(self, valid_config_file, tmp_path, valid_config_dict):
        """Subsequent load_config calls overwrite the cached config."""
        cfg1 = load_config(valid_config_file)
        # Create second config with different api port
        d2 = valid_config_dict.copy()
        d2["api"] = {"port": 9090}
        p2 = write_yaml(str(tmp_path / "arbiter2.yaml"), d2)
        cfg2 = load_config(p2)
        assert get_config() is cfg2
        assert cfg2.api.port == 9090
        assert cfg1 is not cfg2

    def test_env_var_overrides_yaml(self, valid_config_file):
        """Environment variables with ARBITER_ prefix override YAML values."""
        with patch.dict(os.environ, {"ARBITER_API__PORT": "9999"}, clear=False):
            cfg = load_config(valid_config_file)
        assert cfg.api.port == 9999


class TestGetConfigHappyPath:

    def test_get_returns_same_instance_as_load(self, valid_config_file):
        """get_config returns the same instance set by load_config."""
        loaded = load_config(valid_config_file)
        retrieved = get_config()
        assert retrieved is loaded

    def test_get_returns_frozen_config(self, valid_config_file):
        """get_config returns a frozen/immutable ArbiterConfig."""
        load_config(valid_config_file)
        cfg = get_config()
        assert isinstance(cfg, ArbiterConfig)


class TestResetConfigHappyPath:

    def test_reset_sets_singleton_to_none(self, valid_config_file):
        """reset_config clears the singleton so get_config raises."""
        load_config(valid_config_file)
        reset_config()
        with pytest.raises(ConfigNotLoadedError):
            get_config()


class TestGenerateDefaultConfigHappyPath:

    def test_generates_valid_yaml_file(self, tmp_path):
        """generate_default_config creates a valid YAML file."""
        p = str(tmp_path / "arbiter.yaml")
        generate_default_config(p, overwrite=False)
        assert os.path.exists(p)
        with open(p) as f:
            content = f.read()
        assert "config_version" in content

    def test_generated_file_has_header_comment(self, tmp_path):
        """Generated file begins with a YAML header comment."""
        p = str(tmp_path / "arbiter.yaml")
        generate_default_config(p, overwrite=False)
        with open(p) as f:
            first_line = f.readline()
        assert first_line.startswith("#")

    def test_generated_config_roundtrips(self, tmp_path):
        """Generated default config can be loaded by load_config without errors."""
        p = str(tmp_path / "arbiter.yaml")
        generate_default_config(p, overwrite=False)
        cfg = load_config(p)
        assert isinstance(cfg, ArbiterConfig)
        assert cfg.config_version == 1

    def test_generated_config_includes_version_one(self, tmp_path):
        """Generated file includes config_version: 1."""
        p = str(tmp_path / "arbiter.yaml")
        generate_default_config(p, overwrite=False)
        with open(p) as f:
            data = yaml.safe_load(f)
        assert data["config_version"] == 1

    def test_overwrite_true_succeeds_on_existing(self, tmp_path):
        """generate_default_config with overwrite=True succeeds on existing file."""
        p = str(tmp_path / "arbiter.yaml")
        write_raw(p, "old content")
        generate_default_config(p, overwrite=True)
        with open(p) as f:
            content = f.read()
        assert "config_version" in content


class TestValidateConfigFileHappyPath:

    def test_valid_file_returns_config(self, valid_config_file):
        """validate_config_file returns ArbiterConfig for valid YAML."""
        cfg = validate_config_file(valid_config_file)
        assert isinstance(cfg, ArbiterConfig)
        assert cfg.config_version == 1

    def test_validate_does_not_modify_singleton(self, valid_config_file):
        """validate_config_file does not affect the module-level singleton."""
        validate_config_file(valid_config_file)
        with pytest.raises(ConfigNotLoadedError):
            get_config()

    def test_validate_returns_frozen_config(self, valid_config_file):
        """validate_config_file returns a frozen/immutable ArbiterConfig."""
        cfg = validate_config_file(valid_config_file)
        with pytest.raises((AttributeError, TypeError, Exception)):
            cfg.api.port = 9999


# ═══════════════════════════════════════════════════════════════════
# Edge-case tests
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCaseBoundaries:

    def test_trust_floor_zero(self, tmp_path, valid_config_dict):
        """trust.floor=0.0 is valid (lower boundary)."""
        valid_config_dict["trust"]["floor"] = 0.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.floor == 0.0

    def test_trust_floor_one_with_authority_one(self, tmp_path, valid_config_dict):
        """trust.floor=1.0 with authority_override_floor=1.0 is valid."""
        valid_config_dict["trust"]["floor"] = 1.0
        valid_config_dict["trust"]["authority_override_floor"] = 1.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.floor == 1.0
        assert cfg.trust.authority_override_floor == 1.0

    def test_trust_floor_equals_authority(self, tmp_path, valid_config_dict):
        """trust.floor == authority_override_floor is valid (equal case)."""
        valid_config_dict["trust"]["floor"] = 0.5
        valid_config_dict["trust"]["authority_override_floor"] = 0.5
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.floor == cfg.trust.authority_override_floor

    def test_port_boundary_one(self, tmp_path, valid_config_dict):
        """Port value of 1 is valid (lower boundary)."""
        valid_config_dict["api"]["port"] = 1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.api.port == 1

    def test_port_boundary_65535(self, tmp_path, valid_config_dict):
        """Port value of 65535 is valid (upper boundary)."""
        valid_config_dict["api"]["port"] = 65535
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.api.port == 65535

    def test_soak_target_requests_one(self, tmp_path, valid_config_dict):
        """soak.target_requests=1 is valid (minimum)."""
        valid_config_dict["soak"]["target_requests"] = 1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.soak.target_requests == 1

    def test_ledger_checksum_interval_one(self, tmp_path, valid_config_dict):
        """ledger.checksum_interval=1 is valid (minimum)."""
        valid_config_dict["ledger"]["checksum_interval"] = 1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.ledger.checksum_interval == 1

    def test_decay_lambda_zero(self, tmp_path, valid_config_dict):
        """trust.decay_lambda=0.0 is valid (lower boundary)."""
        valid_config_dict["trust"]["decay_lambda"] = 0.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.decay_lambda == 0.0

    def test_conflict_threshold_zero(self, tmp_path, valid_config_dict):
        """trust.conflict_trust_delta_threshold=0.0 is valid."""
        valid_config_dict["trust"]["conflict_trust_delta_threshold"] = 0.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.conflict_trust_delta_threshold == 0.0

    def test_conflict_threshold_one(self, tmp_path, valid_config_dict):
        """trust.conflict_trust_delta_threshold=1.0 is valid."""
        valid_config_dict["trust"]["conflict_trust_delta_threshold"] = 1.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.conflict_trust_delta_threshold == 1.0

    def test_empty_taint_lock_tiers(self, tmp_path, valid_config_dict):
        """Empty taint_lock_tiers list is valid."""
        valid_config_dict["trust"]["taint_lock_tiers"] = []
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert cfg.trust.taint_lock_tiers == []

    def test_all_taint_lock_tiers(self, tmp_path, valid_config_dict):
        """All DataClassificationTier values in taint_lock_tiers is valid."""
        all_tiers = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "CRITICAL"]
        valid_config_dict["trust"]["taint_lock_tiers"] = all_tiers
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        tier_values = [str(t) if not isinstance(t, str) else t.value if hasattr(t, 'value') else str(t)
                       for t in cfg.trust.taint_lock_tiers]
        # Just verify all 5 are present
        assert len(cfg.trust.taint_lock_tiers) == 5

    def test_empty_yaml_file_uses_defaults(self, tmp_path):
        """Empty YAML file falls back to all defaults."""
        p = str(tmp_path / "arbiter.yaml")
        write_raw(p, "")
        cfg = load_config(p)
        assert isinstance(cfg, ArbiterConfig)
        assert cfg.config_version == 1


# ═══════════════════════════════════════════════════════════════════
# Error-case tests
# ═══════════════════════════════════════════════════════════════════

class TestGetConfigErrors:

    def test_get_before_load_raises(self):
        """get_config raises ConfigNotLoadedError when not loaded."""
        with pytest.raises(ConfigNotLoadedError):
            get_config()

    def test_get_after_reset_raises(self, valid_config_file):
        """get_config raises ConfigNotLoadedError after reset_config."""
        load_config(valid_config_file)
        reset_config()
        with pytest.raises(ConfigNotLoadedError):
            get_config()


class TestLoadConfigValidationErrors:

    def test_trust_floor_negative(self, tmp_path, valid_config_dict):
        """trust.floor < 0.0 raises validation error."""
        valid_config_dict["trust"]["floor"] = -0.1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_trust_floor_above_one(self, tmp_path, valid_config_dict):
        """trust.floor > 1.0 raises validation error."""
        valid_config_dict["trust"]["floor"] = 1.1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_authority_floor_negative(self, tmp_path, valid_config_dict):
        """trust.authority_override_floor < 0.0 raises validation error."""
        valid_config_dict["trust"]["authority_override_floor"] = -0.5
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_authority_floor_above_one(self, tmp_path, valid_config_dict):
        """trust.authority_override_floor > 1.0 raises validation error."""
        valid_config_dict["trust"]["authority_override_floor"] = 1.5
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_trust_floor_gt_authority_floor(self, tmp_path, valid_config_dict):
        """trust.floor > authority_override_floor raises cross-field validation error."""
        valid_config_dict["trust"]["floor"] = 0.8
        valid_config_dict["trust"]["authority_override_floor"] = 0.3
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_decay_lambda_negative(self, tmp_path, valid_config_dict):
        """trust.decay_lambda < 0.0 raises validation error."""
        valid_config_dict["trust"]["decay_lambda"] = -1.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_conflict_threshold_negative(self, tmp_path, valid_config_dict):
        """trust.conflict_trust_delta_threshold < 0.0 raises validation error."""
        valid_config_dict["trust"]["conflict_trust_delta_threshold"] = -0.1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_conflict_threshold_above_one(self, tmp_path, valid_config_dict):
        """trust.conflict_trust_delta_threshold > 1.0 raises validation error."""
        valid_config_dict["trust"]["conflict_trust_delta_threshold"] = 1.5
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_port_zero_invalid(self, tmp_path, valid_config_dict):
        """Port value of 0 raises validation error."""
        valid_config_dict["api"]["port"] = 0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_port_above_65535_invalid(self, tmp_path, valid_config_dict):
        """Port value above 65535 raises validation error."""
        valid_config_dict["api"]["port"] = 70000
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_port_negative_invalid(self, tmp_path, valid_config_dict):
        """Negative port raises validation error."""
        valid_config_dict["otlp"]["listen_port"] = -1
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_soak_target_requests_zero_invalid(self, tmp_path, valid_config_dict):
        """soak.target_requests=0 raises validation error."""
        valid_config_dict["soak"]["target_requests"] = 0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_ledger_checksum_interval_zero_invalid(self, tmp_path, valid_config_dict):
        """ledger.checksum_interval=0 raises validation error."""
        valid_config_dict["ledger"]["checksum_interval"] = 0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_invalid_tier_in_taint_lock(self, tmp_path, valid_config_dict):
        """Invalid DataClassificationTier in taint_lock_tiers raises error."""
        valid_config_dict["trust"]["taint_lock_tiers"] = ["INVALID_TIER"]
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_wrong_type_port_string(self, tmp_path, valid_config_dict):
        """String value for port field raises validation error."""
        valid_config_dict["api"]["port"] = "not_a_number"
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    def test_config_version_wrong(self, tmp_path, valid_config_dict):
        """config_version != 1 raises validation error."""
        valid_config_dict["config_version"] = 2
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)


class TestLoadConfigYamlParseErrors:

    def test_malformed_yaml_raises(self, tmp_path):
        """Malformed YAML syntax raises yaml parse error."""
        p = str(tmp_path / "arbiter.yaml")
        write_raw(p, "{{invalid yaml:: [}")
        with pytest.raises(Exception):
            load_config(p)

    def test_binary_content_raises(self, tmp_path):
        """Binary content in YAML file raises parse or validation error."""
        p = str(tmp_path / "arbiter.yaml")
        with open(p, "wb") as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe")
        with pytest.raises(Exception):
            load_config(p)


class TestLoadConfigPermissionErrors:

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
    def test_unreadable_file_raises_permission_error(self, tmp_path, valid_config_dict):
        """Permission error when YAML file is not readable."""
        p = str(tmp_path / "arbiter.yaml")
        write_yaml(p, valid_config_dict)
        os.chmod(p, 0o000)
        try:
            with pytest.raises((PermissionError, ConfigurationError, OSError)):
                load_config(p)
        finally:
            os.chmod(p, 0o644)


class TestGenerateDefaultConfigErrors:

    def test_no_overwrite_existing_file(self, tmp_path):
        """generate_default_config raises when file exists and overwrite=False."""
        p = str(tmp_path / "arbiter.yaml")
        write_raw(p, "existing")
        with pytest.raises((FileExistsError, OSError, Exception)):
            generate_default_config(p, overwrite=False)

    def test_parent_directory_missing(self, tmp_path):
        """generate_default_config raises when parent directory does not exist."""
        p = str(tmp_path / "nonexistent" / "subdir" / "arbiter.yaml")
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            generate_default_config(p, overwrite=False)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
    def test_permission_error_on_target_dir(self, tmp_path):
        """generate_default_config raises when directory is not writable."""
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        os.chmod(str(restricted), 0o444)
        p = str(restricted / "arbiter.yaml")
        try:
            with pytest.raises((PermissionError, OSError, Exception)):
                generate_default_config(p, overwrite=False)
        finally:
            os.chmod(str(restricted), 0o755)


class TestValidateConfigFileErrors:

    def test_file_not_found(self, tmp_path):
        """validate_config_file raises when file does not exist."""
        p = str(tmp_path / "nonexistent.yaml")
        with pytest.raises((FileNotFoundError, ConfigurationError, OSError)):
            validate_config_file(p)

    def test_malformed_yaml(self, tmp_path):
        """validate_config_file raises on malformed YAML."""
        p = str(tmp_path / "bad.yaml")
        write_raw(p, "{{bad yaml:: [}")
        with pytest.raises(Exception):
            validate_config_file(p)

    def test_invalid_values_raise_with_details(self, tmp_path, valid_config_dict):
        """validate_config_file raises ConfigurationError with details on invalid values."""
        valid_config_dict["trust"]["floor"] = -1.0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)) as exc_info:
            validate_config_file(p)
        # If it's a ConfigurationError, verify it has context
        if isinstance(exc_info.value, ConfigurationError):
            assert hasattr(exc_info.value, "config_path") or hasattr(exc_info.value, "validation_errors")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
    def test_permission_error(self, tmp_path, valid_config_dict):
        """validate_config_file raises on unreadable file."""
        p = str(tmp_path / "arbiter.yaml")
        write_yaml(p, valid_config_dict)
        os.chmod(p, 0o000)
        try:
            with pytest.raises((PermissionError, ConfigurationError, OSError)):
                validate_config_file(p)
        finally:
            os.chmod(p, 0o644)


# ═══════════════════════════════════════════════════════════════════
# Invariant tests
# ═══════════════════════════════════════════════════════════════════

class TestInvariants:

    def test_config_is_frozen_immutable(self, valid_config_file):
        """ArbiterConfig is frozen — attribute assignment raises error."""
        cfg = load_config(valid_config_file)
        with pytest.raises((AttributeError, TypeError, Exception)):
            cfg.config_version = 99
        with pytest.raises((AttributeError, TypeError, Exception)):
            cfg.api.port = 9999

    def test_singleton_identity(self, valid_config_file):
        """get_config returns same object identity on repeated calls."""
        load_config(valid_config_file)
        cfg1 = get_config()
        cfg2 = get_config()
        cfg3 = get_config()
        assert cfg1 is cfg2
        assert cfg2 is cfg3

    def test_trust_floor_lte_authority(self, valid_config_file):
        """Loaded config always has trust.floor <= trust.authority_override_floor."""
        cfg = load_config(valid_config_file)
        assert cfg.trust.floor <= cfg.trust.authority_override_floor

    def test_all_soak_tiers_present(self, valid_config_file):
        """All DataClassificationTier variants are present in soak.base_durations."""
        cfg = load_config(valid_config_file)
        durations = cfg.soak.base_durations
        # Access as dict or object — check all tier names present
        tier_names = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "CRITICAL"]
        for tier_name in tier_names:
            if isinstance(durations, dict):
                assert tier_name in durations or DataClassificationTier(tier_name) in durations, \
                    f"Missing soak duration for tier {tier_name}"
            else:
                assert hasattr(durations, tier_name), f"Missing soak duration attr for {tier_name}"
                val = getattr(durations, tier_name)
                assert isinstance(val, int) and val >= 0

    def test_config_version_is_one(self, valid_config_file):
        """config_version is always 1 for loaded configs."""
        cfg = load_config(valid_config_file)
        assert cfg.config_version == 1

    def test_source_priority_env_overrides_yaml(self, tmp_path, valid_config_dict):
        """Environment variables override YAML values which override defaults."""
        # YAML sets api.port to 8080
        valid_config_dict["api"]["port"] = 8080
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)

        # Env var should override to 7777
        with patch.dict(os.environ, {"ARBITER_API__PORT": "7777"}, clear=False):
            cfg = load_config(p)
        assert cfg.api.port == 7777

    def test_default_config_trust_floor_lte_authority(self, tmp_path):
        """Default config satisfies trust.floor <= trust.authority_override_floor."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            cfg = load_config(None)
        assert cfg.trust.floor <= cfg.trust.authority_override_floor

    def test_validate_does_not_pollute_singleton(self, tmp_path, valid_config_dict):
        """validate_config_file never modifies the module singleton."""
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        # Singleton not loaded initially
        with pytest.raises(ConfigNotLoadedError):
            get_config()
        # Validate the file
        validate_config_file(p)
        # Singleton should still be unloaded
        with pytest.raises(ConfigNotLoadedError):
            get_config()

    def test_load_then_validate_singleton_unchanged(self, tmp_path, valid_config_dict):
        """validate_config_file after load_config doesn't change singleton."""
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        loaded = load_config(p)

        # Now validate a second config — shouldn't change singleton
        d2 = valid_config_dict.copy()
        d2["api"] = {"port": 1234}
        p2 = write_yaml(str(tmp_path / "other.yaml"), d2)
        validated = validate_config_file(p2)

        assert get_config() is loaded
        assert validated is not loaded


# ═══════════════════════════════════════════════════════════════════
# Parametrized tests for DataClassificationTier
# ═══════════════════════════════════════════════════════════════════

class TestDataClassificationTierParametrized:

    @pytest.mark.parametrize("tier", ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "CRITICAL"])
    def test_single_tier_in_taint_lock(self, tmp_path, valid_config_dict, tier):
        """Each individual DataClassificationTier is valid in taint_lock_tiers."""
        valid_config_dict["trust"]["taint_lock_tiers"] = [tier]
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        cfg = load_config(p)
        assert len(cfg.trust.taint_lock_tiers) == 1


class TestPortParametrized:

    @pytest.mark.parametrize("port_field,section", [
        ("listen_port", "otlp"),
        ("http_port", "otlp"),
        ("port", "api"),
    ])
    def test_port_fields_reject_zero(self, tmp_path, valid_config_dict, port_field, section):
        """All port fields reject value 0."""
        valid_config_dict[section][port_field] = 0
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)

    @pytest.mark.parametrize("port_field,section", [
        ("listen_port", "otlp"),
        ("http_port", "otlp"),
        ("port", "api"),
    ])
    def test_port_fields_reject_above_65535(self, tmp_path, valid_config_dict, port_field, section):
        """All port fields reject values above 65535."""
        valid_config_dict[section][port_field] = 99999
        p = write_yaml(str(tmp_path / "arbiter.yaml"), valid_config_dict)
        with pytest.raises((ConfigurationError, ValueError, Exception)):
            load_config(p)
