# -*- coding: utf-8 -*-
"""auth module tests: password hashing, JWT, register/login API, DB ops."""
import pytest, time
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from auth.service import hash_password, verify_password, create_token, verify_token
from auth.db import init_db, create_user, get_user_by_username, count_users
from auth.router import router as auth_router
import config as C

class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "secure_pw_123"
        hashed = hash_password(pw)
        assert hashed != pw
        assert verify_password(pw, hashed) is True

    def test_wrong_password_fails(self):
        assert verify_password("wrong", hash_password("correct")) is False

    def test_different_salts(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
        assert verify_password("same", h1) and verify_password("same", h2)

class TestJWT:
    def test_create_and_verify(self):
        token = create_token("alice", "user")
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "alice"
        assert payload["role"] == "user"

    def test_invalid_token(self):
        assert verify_token("invalid.token.here") is None

    def test_tampered_token(self):
        token = create_token("alice", "user")
        assert verify_token(token[:-5] + "XXXXX") is None

    def test_expired_token(self):
        # Craft a JWT that expired 1 hour ago
        import jwt as _jwt
        now = int(time.time())
        payload = {"sub": "alice", "role": "user", "iat": now - 90000, "exp": now - 3600}
        token = _jwt.encode(payload, C.JWT_SECRET, algorithm=C.JWT_ALGORITHM)
        assert verify_token(token) is None
class TestDBOperations:
    def test_create_and_get_user(self, tmp_auth_db):
        h = hash_password("pw123456")
        assert create_user("testuser", h, "user") is True
        user = get_user_by_username("testuser")
        assert user is not None
        assert user["username"] == "testuser"
        assert user["role"] == "user"

    def test_duplicate_user_rejected(self, tmp_auth_db):
        h = hash_password("pw123456")
        assert create_user("dup", h, "user") is True
        assert create_user("dup", h, "user") is False

    def test_get_nonexistent_user(self, tmp_auth_db):
        assert get_user_by_username("ghost") is None

    def test_count_users(self, tmp_auth_db):
        assert count_users() == 0
        create_user("u1", hash_password("pw123456"), "user")
        assert count_users() == 1
        create_user("u2", hash_password("pw123456"), "user")
        assert count_users() == 2

    def test_init_db_idempotent(self, tmp_auth_db):
        init_db(); init_db()
        assert count_users() == 0

def _make_auth_client():
    app = FastAPI()
    app.include_router(auth_router, prefix="/api/auth")
    return TestClient(app)

class TestRegisterAPI:
    def test_first_user_admin(self, tmp_auth_db):
        c = _make_auth_client()
        r = c.post("/api/auth/register", json={"username": "admin", "password": "password123"})
        assert r.status_code == 200
        d = r.json()
        assert d["username"] == "admin" and d["role"] == "admin" and "token" in d

    def test_second_user_regular(self, tmp_auth_db):
        c = _make_auth_client()
        c.post("/api/auth/register", json={"username": "admin", "password": "pw123456"})
        r = c.post("/api/auth/register", json={"username": "user1", "password": "pw123456"})
        assert r.status_code == 200 and r.json()["role"] == "user"

    def test_duplicate_rejected(self, tmp_auth_db):
        c = _make_auth_client()
        c.post("/api/auth/register", json={"username": "dup", "password": "pw123456"})
        r = c.post("/api/auth/register", json={"username": "dup", "password": "pw123456"})
        assert r.status_code == 409

    def test_short_username(self, tmp_auth_db):
        c = _make_auth_client()
        assert c.post("/api/auth/register", json={"username": "ab", "password": "pw123456"}).status_code == 400

    def test_long_username(self, tmp_auth_db):
        c = _make_auth_client()
        assert c.post("/api/auth/register", json={"username": "a"*21, "password": "pw123456"}).status_code == 400

    def test_short_password(self, tmp_auth_db):
        c = _make_auth_client()
        assert c.post("/api/auth/register", json={"username": "valid", "password": "12345"}).status_code == 400

    def test_special_chars_username(self, tmp_auth_db):
        c = _make_auth_client()
        assert c.post("/api/auth/register", json={"username": "user@name", "password": "pw123456"}).status_code == 400

class TestLoginAPI:
    def test_login_correct(self, tmp_auth_db):
        c = _make_auth_client()
        c.post("/api/auth/register", json={"username": "loginuser", "password": "pw123456"})
        r = c.post("/api/auth/login", json={"username": "loginuser", "password": "pw123456"})
        assert r.status_code == 200 and "token" in r.json()

    def test_login_wrong_password(self, tmp_auth_db):
        c = _make_auth_client()
        c.post("/api/auth/register", json={"username": "loginuser", "password": "correct"})
        assert c.post("/api/auth/login", json={"username": "loginuser", "password": "wrong"}).status_code == 401

    def test_login_nonexistent(self, tmp_auth_db):
        c = _make_auth_client()
        r = c.post("/api/auth/login", json={"username": "ghost", "password": "any"})
        assert r.status_code == 401

class TestMeAPI:
    def test_me_valid_token(self, tmp_auth_db):
        c = _make_auth_client()
        reg = c.post("/api/auth/register", json={"username": "meuser", "password": "pw123456"})
        r = c.get("/api/auth/me", headers={"Authorization": f"Bearer {reg.json()['token']}"})
        assert r.status_code == 200 and r.json()["username"] == "meuser"

    def test_me_no_token(self, tmp_auth_db):
        c = _make_auth_client()
        assert c.get("/api/auth/me").status_code == 401

    def test_me_invalid_token(self, tmp_auth_db):
        c = _make_auth_client()
        assert c.get("/api/auth/me", headers={"Authorization": "Bearer bad"}).status_code == 401
