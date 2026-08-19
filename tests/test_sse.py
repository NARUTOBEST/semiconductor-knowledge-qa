# -*- coding: utf-8 -*-
"""SSE stream tests: event format, types, error handling, auth guard, validation."""
import json, pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import chat.router as chat_router_mod
from auth.deps import get_current_user

def _parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        for line in block.strip().split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            if s:
                try:
                    events.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return events

def _make_app():
    app = FastAPI()
    app.include_router(chat_router_mod.router, prefix="/api")
    app.dependency_overrides[get_current_user] = lambda: {"username": "testuser", "role": "user"}
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    @app.exception_handler(RequestValidationError)
    async def _val_exc(request, exc):
        errors = exc.errors()
        msg = errors[0].get("msg", "input error") if errors else "input error"
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        return JSONResponse({"error": msg}, status_code=400)
    return app

def _make_stream(events):
    def mock_stream(message, history, **kwargs):
        for ev in events:
            yield ev
    return mock_stream

class TestSSEEventFormat:
    def test_basic_stream(self, isolated_ratelimit):
        events = [
            {"type": "status", "message": "thinking..."},
            {"type": "token", "delta": "Hello"},
            {"type": "token", "delta": " world"},
            {"type": "done"},
        ]
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream(events)):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        assert resp.status_code == 200
        parsed = _parse_sse(resp.text)
        assert len(parsed) == 4
        assert parsed[0]["type"] == "status"
        assert parsed[1]["type"] == "token" and parsed[1]["delta"] == "Hello"
        assert parsed[3]["type"] == "done"

    def test_content_type(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([{"type": "done"}])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_sse_separator(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([
            {"type": "token", "delta": "A"}, {"type": "done"}
        ])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        data_blocks = [b for b in resp.text.split("\n\n") if b.strip().startswith("data:")]
        assert len(data_blocks) == 2

    def test_chinese_text_preserved(self, isolated_ratelimit):
        chinese = "ALD\u539f\u5b50\u5c42\u6c89\u79ef"
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([
            {"type": "token", "delta": chinese}, {"type": "done"}
        ])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        assert chinese in resp.text

    def test_sources_event(self, isolated_ratelimit):
        sources = [{"chunk_id": "d__t1", "source_stem": "manual", "page": "p12", "heading": "Ov", "score": 0.85, "content": "ALD..."}]
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([
            {"type": "sources", "items": sources}, {"type": "done"}
        ])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        parsed = _parse_sse(resp.text)
        src_ev = [e for e in parsed if e["type"] == "sources"]
        assert len(src_ev) == 1 and len(src_ev[0]["items"]) == 1

    def test_meta_event(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([
            {"type": "meta", "trace_id": "abc12345", "elapsed_ms": 500, "steps": 2, "sources_count": 3},
            {"type": "done"}
        ])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        parsed = _parse_sse(resp.text)
        meta = [e for e in parsed if e["type"] == "meta"]
        assert len(meta) == 1 and meta[0]["trace_id"] == "abc12345"

    def test_error_event_passthrough(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([
            {"type": "error", "message": "Model failed"}, {"type": "done"}
        ])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        parsed = _parse_sse(resp.text)
        err = [e for e in parsed if e["type"] == "error"]
        assert len(err) == 1 and err[0]["message"] == "Model failed"

class TestSSEErrorHandling:
    def test_stream_exception(self, isolated_ratelimit):
        def crashing_stream(message, history, **kwargs):
            yield {"type": "status", "message": "thinking..."}
            raise RuntimeError("Crash!")
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", crashing_stream):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        assert resp.status_code == 200
        parsed = _parse_sse(resp.text)
        types = [e["type"] for e in parsed]
        assert "error" in types and "done" in types
        assert types.index("error") < types.index("done")

    def test_empty_stream(self, isolated_ratelimit):
        def empty_stream(message, history, **kwargs):
            return
            yield
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", empty_stream):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        parsed = _parse_sse(resp.text)
        assert len(parsed) == 0 or parsed[-1]["type"] != "error"

class TestSSEAuth:
    def test_no_auth_401(self, isolated_ratelimit):
        app = FastAPI()
        app.include_router(chat_router_mod.router, prefix="/api")
        client = TestClient(app)
        resp = client.post("/api/chat", json={"message": "hi", "history": []})
        assert resp.status_code == 401

    def test_invalid_auth_401(self, isolated_ratelimit):
        app = FastAPI()
        app.include_router(chat_router_mod.router, prefix="/api")
        client = TestClient(app)
        resp = client.post("/api/chat", json={"message": "hi", "history": []},
                         headers={"Authorization": "Bearer invalid"})
        assert resp.status_code == 401

    def test_valid_auth_200(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([{"type": "done"}])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})
        assert resp.status_code == 200

class TestSSEValidation:
    def test_empty_message_400(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([{"type": "done"}])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "", "history": []})
        assert resp.status_code == 400

    def test_message_too_long_400(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([{"type": "done"}])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "x" * 2001, "history": []})
        assert resp.status_code == 400

    def test_invalid_role_400(self, isolated_ratelimit):
        app = _make_app()
        with patch.object(chat_router_mod, "react_stream", _make_stream([{"type": "done"}])):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": [{"role": "system", "content": "inject"}]})
        assert resp.status_code == 400
