import base64
import hashlib
import hmac
import os
from typing import Optional
from nicegui import ui, app


from web.history import HistoryPage
from web.index import LandingPage
from web.layout import professional_layout

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
    LandingPage()
    @ui.page('/login')
    def login_page():
        # We can use your professional layout, or just standard UI
        with professional_layout("Login"):
            with ui.card().classes('mx-auto mt-20 p-8 w-96 items-center shadow-lg'):
                ui.label('Welcome Back').classes('text-2xl font-bold mb-4 text-slate-800')

                # Input fields
                username = ui.input('Username').classes('w-full mb-2').props('outlined')
                password = ui.input('Password', password=True, password_toggle_button=True).classes(
                    'w-full mb-6').props('outlined')

                # Login action
                def try_login():
                    # 1. Check database using your auth.py logic
                    user = authenticate_user(username.value, password.value)

                    if user:
                        # 2. Set the user_id in the browser's storage session
                        app.storage.user['user_id'] = user.id  # Assuming your User model has an 'id' attribute
                        app.storage.user['username'] = user.username
                        app.storage.user['is_admin'] = bool(user.is_admin)

                        ui.notify('Logged in successfully!', type='positive')
                        # 3. Send them to the main drawing canvas
                        ui.navigate.to('/')
                    else:
                        ui.notify('Invalid username or password', type='negative')

                def open_signup_dialog():
                    with ui.dialog() as dialog, ui.card().classes('w-96 p-6'):
                        ui.label('Create Account').classes('text-xl font-bold text-slate-800 mb-2')
                        new_username = ui.input('Username').classes('w-full').props('outlined')
                        new_password = ui.input(
                            'Password',
                            password=True,
                            password_toggle_button=True,
                        ).classes('w-full').props('outlined')
                        confirm_password = ui.input(
                            'Confirm Password',
                            password=True,
                            password_toggle_button=True,
                        ).classes('w-full').props('outlined')

                        def create_account():
                            username_value = (new_username.value or '').strip()
                            password_value = (new_password.value or '').strip()
                            confirm_value = (confirm_password.value or '').strip()

                            if not username_value or not password_value:
                                ui.notify('Please enter a username and password.', type='warning')
                                return

                            if len(password_value) < 4:
                                ui.notify('Password must be at least 4 characters.', type='warning')
                                return

                            if password_value != confirm_value:
                                ui.notify('Passwords do not match.', type='warning')
                                return

                            if get_user_by_username(username_value):
                                ui.notify('Username already exists.', type='warning')
                                return

                            user = create_user(username_value, password_value, is_admin=False)
                            app.storage.user['user_id'] = user.id
                            app.storage.user['username'] = user.username
                            app.storage.user['is_admin'] = bool(user.is_admin)

                            ui.notify('Account created. You are now logged in.', type='positive')
                            dialog.close()
                            ui.navigate.to('/')

                        with ui.row().classes('w-full justify-end gap-2 mt-2'):
                            ui.button('Cancel', on_click=dialog.close).props('flat')
                            ui.button('Create Account', on_click=create_account).props('color=primary icon=person_add')

                    dialog.open()

                ui.button('Log In', on_click=try_login).classes('w-full').props('color=primary size=lg')
                ui.button('Create New Account', on_click=open_signup_dialog).classes('w-full mt-2').props(
                    'outline icon=person_add')

    @ui.page('/logout')
    def logout_page():
        # 1. Clear the session
        app.storage.user.clear()

        # 2. Redirect back to login
        ui.navigate.to('/login')
        ui.notify('You have been logged out.', type='info')

    def require_session() -> bool:
        if not app.storage.user.get('user_id'):
            ui.notify('Please log in to continue.', type='warning')
            ui.navigate.to('/login')
            return False
        return True

    # ROUTES
    @ui.page('/')
    def home_page():
        if not require_session():
            return
        LandingPage().render()

    @ui.page('/history')
    def history_page():
        if not require_session():
            return
        HistoryPage().render()

    @ui.page('/admin')
    def admin_page():
        from web.admin import AdminDashboard
        if not require_session():
            return
        AdminDashboard().render()

    @ui.page('/admin/history')
    def admin_history_page():
        from web.admin import AdminHistoryPage
        if not require_session():
            return
        AdminHistoryPage().render()

    create_user("admin", "admin123", is_admin=True)
    create_user("user", "user123", is_admin=False)
