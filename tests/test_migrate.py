# -*- coding: utf-8 -*-
"""migrate.run_migrations 的单测:用 fake cursor/connection 验证版本追踪与幂等。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memories" / "db"))

import migrate  # noqa: E402


class _FakeCursor:
    def __init__(self, executed, fetchone_results):
        self._executed = executed
        self._fetchone_results = list(fetchone_results)
        self._rows = []

    def execute(self, sql, params=None):
        self._executed.append((sql.strip(), params))
        low = sql.strip().lower()
        if low.startswith("select version from schema_migrations"):
            self._rows = [("001",)]
        elif "create table" in low and "schema_migrations" in low:
            self._rows = []
        else:
            self._rows = []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, executed):
        self.autocommit = True
        self._executed = executed

    def cursor(self):
        return _FakeCursor(self._executed, [])

    def commit(self):
        self._executed.append(("COMMIT", None))

    def rollback(self):
        self._executed.append(("ROLLBACK", None))

    def close(self):
        pass


def test_discover_parses_header_and_sorts(tmp_path):
    (tmp_path / "002_b.sql").write_text("-- database: agent_long_db\nSELECT 2;",
                                        encoding="utf-8")
    (tmp_path / "001_a.sql").write_text("-- database: agent_short_db\nSELECT 1;",
                                        encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    migrations = migrate.discover_migrations(str(tmp_path))
    assert [v for v, _, _ in migrations] == ["001", "002"]
    assert migrations[0][1] == "agent_short_db"
    assert migrations[1][1] == "agent_long_db"


def test_missing_database_header_raises(tmp_path):
    (tmp_path / "001_bad.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="database"):
        migrate.discover_migrations(str(tmp_path))


def test_applied_version_is_skipped(monkeypatch, tmp_path):
    (tmp_path / "001_a.sql").write_text("-- database: agent_short_db\nSELECT 1;",
                                        encoding="utf-8")
    executed = []

    monkeypatch.setattr(migrate, "_ensure_version_table", lambda uri: None)
    monkeypatch.setattr(migrate, "_applied_versions",
                        lambda uri: {"001"} if uri == "short" else set())

    applied = migrate.run_migrations({"agent_short_db": "short"},
                                     directory=str(tmp_path),
                                     printer=lambda *a, **k: None)
    assert applied == []


def test_unknown_db_raises(monkeypatch, tmp_path):
    (tmp_path / "001_a.sql").write_text("-- database: ghost_db\nSELECT 1;",
                                        encoding="utf-8")
    monkeypatch.setattr(migrate, "_ensure_version_table", lambda uri: None)
    monkeypatch.setattr(migrate, "_applied_versions", lambda uri: set())
    with pytest.raises(KeyError):
        migrate.run_migrations({"agent_short_db": "short"},
                               directory=str(tmp_path),
                               printer=lambda *a, **k: None)
