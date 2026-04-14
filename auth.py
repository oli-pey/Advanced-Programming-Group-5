import base64
import hashlib
import hmac
import os
from typing import Optional

from nicegui import ui, app
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


def require_session() -> bool:
    if not app.storage.user.get('user_id'):
        ui.notify('Please log in to continue.', type='warning')
        ui.navigate.to('/login')
        return False
    return True