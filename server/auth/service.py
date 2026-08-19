# -*- coding: utf-8 -*-
"""认证核心逻辑:密码哈希(bcrypt)+ JWT 签发/验证。"""
import time
import bcrypt
import jwt
import config as C


def hash_password(password: str) -> str:
    """bcrypt 哈希密码,返回 str(可安全存入 DB)。"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """验证明文密码是否匹配哈希。"""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(username: str, role: str) -> str:
    """签发 JWT,有效期 JWT_EXPIRE_HOURS 小时。"""
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + C.JWT_EXPIRE_HOURS * 3600,
    }
    return jwt.encode(payload, C.JWT_SECRET, algorithm=C.JWT_ALGORITHM)


def verify_token(token: str) -> dict | None:
    """验证 JWT,有效返回 payload dict,无效/过期返回 None。"""
    try:
        return jwt.decode(token, C.JWT_SECRET, algorithms=[C.JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
