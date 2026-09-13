"""Give each consistency case an isolated store with a real file boundary."""

import pytest
import arbiter.consistency as consistency
from arbiter.consistency import FindingStore


@pytest.fixture(autouse=True)
def isolated_finding_store(tmp_path, monkeypatch):
    monkeypatch.setattr(consistency, "_default_store", FindingStore(tmp_path / "findings.jsonl"))
