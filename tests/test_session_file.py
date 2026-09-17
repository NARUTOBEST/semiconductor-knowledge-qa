# -*- coding: utf-8 -*-
"""session-memory.md 文件存储(session_file.py)单测:tmp_path,不触网。

覆盖:路径命名(按用户隔离 + 哈希,scoped id 含 | 安全)、front-matter 往返、
原子写(读方看不到半成品)、文件锁争用/超时、9 章节模板、delete_user_files。
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import config as C  # noqa: E402

from memories.storage.working import session_file as SF  # noqa: E402


@pytest.fixture()
def sess_dir(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    monkeypatch.setattr(C, "MEM_SESSION_DIR", str(d), raising=False)
    return str(d)


def test_path_isolated_and_safe(sess_dir):
    p1 = SF.session_path("alice", "alice|t1")
    p2 = SF.session_path("bob", "bob|t1")
    # 按用户分目录
    assert os.sep + "alice" + os.sep in p1
    assert os.sep + "bob" + os.sep in p2
    assert p1 != p2
    # 文件名是哈希(无 | / 等特殊字符),固定后缀
    name = os.path.basename(p1)
    assert "|" not in name and name.endswith(".session-memory.md")
    # 匿名用户落到 _anonymous
    p3 = SF.session_path(None, "x|t")
    assert os.sep + "_anonymous" + os.sep in p3


def test_frontmatter_roundtrip(sess_dir):
    f = SF.SessionFile("alice", "alice|t1")
    assert not f.exists()
    meta = {"cursor_msg_id": "m-9", "cursor_index": 9, "last_tokens": 12345,
            "last_tool_calls": 4, "version": 1, "updated_at": 1700000000}
    body = "## 1. 会话目标\n做测试\n"
    f.write(meta, body)
    assert f.exists()
    m, b = f.read()
    assert m["cursor_msg_id"] == "m-9"
    assert "做测试" in b
    tm = f.read_meta_typed()
    assert tm["cursor_index"] == 9 and tm["last_tokens"] == 12345
    assert tm["last_tool_calls"] == 4 and tm["version"] == 1


def test_nine_section_template(sess_dir):
    body = SF.empty_template_body()
    for i in range(1, 10):
        assert f"{i}." in body
    assert "会话目标" in body and "下一步建议" in body


def test_lock_contention_and_timeout(sess_dir):
    f = SF.SessionFile("alice", "alice|t1")
    # 同线程重入直接放行(整周期锁内再调 write 不自等超时)
    with SF.file_lock(f.path, timeout=0.3):
        with SF.file_lock(f.path, timeout=0.3):
            pass
    # 跨线程互斥:另一线程持锁时,本线程获取应超时
    got = threading.Event()
    release = threading.Event()

    def holder():
        with SF.file_lock(f.path, timeout=1.0):
            got.set()
            release.wait(2.0)

    t = threading.Thread(target=holder)
    t.start()
    assert got.wait(1.0)
    with pytest.raises(TimeoutError):
        with SF.file_lock(f.path, timeout=0.3):
            pass
    release.set()
    t.join(2.0)
    # 释放后可重新获取
    with SF.file_lock(f.path, timeout=1.0):
        pass


def test_session_lock_wraps_full_cycle(sess_dir):
    """整周期锁:锁内读→写(write 内部重入同一把锁)不死锁;跨线程全程互斥。"""
    f = SF.SessionFile("alice", "alice|t1")
    f.write({"cursor_index": 0}, "v1")
    order = []

    def cycle():
        with f.lock(timeout=2.0):
            meta, body = f.read()            # 锁内读游标
            order.append(f"read:{meta.get('cursor_index')}")
            time.sleep(0.05)
            meta["cursor_index"] = int(meta.get("cursor_index", 0)) + 1
            f.write(meta, body)              # 锁内原子替换(重入)
            order.append(f"write:{meta['cursor_index']}")

    t1 = threading.Thread(target=cycle)
    t2 = threading.Thread(target=cycle)
    t1.start(); t2.start()
    t1.join(3.0); t2.join(3.0)
    # 两个周期严格串行:读到的游标必为 0→1(第二个周期建立在第一个的结果上,不互相覆盖)
    assert order == ["read:0", "write:1", "read:1", "write:2"]
    meta, _ = f.read()
    assert meta.get("cursor_index") == "2"
    assert not os.path.exists(f.path + ".lock")


def test_lock_released_after_write(sess_dir):
    f = SF.SessionFile("alice", "alice|t1")
    f.write({"cursor_index": 1}, "body")
    # 写完锁已释放:可再次获取
    with SF.file_lock(f.path, timeout=1.0):
        pass
    # 无残留 .lock
    assert not os.path.exists(f.path + ".lock")


def test_delete_user_files(sess_dir):
    f1 = SF.SessionFile("alice", "alice|t1")
    f2 = SF.SessionFile("alice", "alice|t2")
    f1.write({"cursor_index": 1}, "a")
    f2.write({"cursor_index": 1}, "b")
    bob = SF.SessionFile("bob", "bob|t1")
    bob.write({"cursor_index": 1}, "c")
    SF.delete_user_files("alice")
    assert not f1.exists() and not f2.exists()
    assert bob.exists()
