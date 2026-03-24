import base64
import hashlib
import hmac
import os
from typing import Optional

from DB.database import SessionLocal, User


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(salt + digest).decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    raw = base64.b64decode(stored_hash.encode("utf-8"))
    salt = raw[:16]
    expected = raw[16:]
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(expected, actual)


def get_user_by_username(username: str) -> Optional[User]:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> Optional[User]:
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_user(username: str, password: str, is_admin: bool = False) -> User:
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return existing

        user = User(
            username=username,
            password_hash=hash_password(password),
            is_admin=is_admin,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def bootstrap_defaults() -> None:
    """
    Creates one default admin and one default demo user if they do not exist yet.
    Change or remove this for production.
    """
    create_user("admin", "admin123", is_admin=True)
    create_user("user", "user123", is_admin=False)
