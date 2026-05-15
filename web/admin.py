import base64

from nicegui import app, ui

import auth
from auth import create_user, get_user_by_username, hash_password
from DB.database import PredictionEntry, SessionLocal, User
from web.layout import professional_layout


def _is_current_user_admin() -> bool:
    """
    Resolve admin status from session first, then database fallback.

    Returns:
        bool: True if the current user is an administrator.
    """
    if app.storage.user.get("is_admin", False):
        return True

    user_id = app.storage.user.get("user_id")
    if not user_id:
        return False

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        is_admin = bool(user and user.is_admin)
        app.storage.user["is_admin"] = is_admin
        return is_admin
    finally:
        db.close()


class AdminDashboard:
    """
    Dashboard for administrative tasks including user creation
    and permission management.
    """

    def render(self):
        """Render the administrative control panel UI."""
        if not _is_current_user_admin():
            ui.notify("Admin access only.", type="negative")
            ui.navigate.to("/")
            return

        username_input = None
        password_input = None
        admin_toggle = None
        password_inputs = {}
        user_management_container = None

        def load_users():
            """Fetch all users ordered by username."""
            db = SessionLocal()
            try:
                return db.query(User).order_by(User.username.asc()).all()
            finally:
                db.close()

        def refresh_user_management() -> None:
            """Refresh the user list display."""
            if not user_management_container:
                return

            users = load_users()
            password_inputs.clear()
            user_management_container.clear()

            with user_management_container:
                if not users:
                    ui.label("No users found.").classes(
                        "text-slate-500 italic"
                    )
                    return

                for user in users:
                    with ui.row().classes(
                        "w-full items-center gap-3 p-3 bg-slate-50 "
                        "rounded-xl border border-slate-200"
                    ):
                        ui.label(user.username).classes(
                            "min-w-32 font-medium text-slate-800"
                        )
                        ui.label("Admin" if user.is_admin else "User").classes(
                            "min-w-16 text-xs uppercase text-slate-500"
                        )

                        admin_switch = ui.switch(
                            "Admin", value=bool(user.is_admin)
                        )
                        admin_switch.on_value_change(
                            lambda e, uid=user.id: update_admin_status(
                                uid, bool(e.value)
                            )
                        )

                        pwd_input = ui.input(
                            f"New password for {user.username}",
                            password=True,
                            password_toggle_button=True,
                        ).props("dense outlined").classes("w-72")
                        password_inputs[user.id] = pwd_input

                        ui.button(
                            "Reset Password",
                            on_click=lambda uid=user.id: reset_password(uid)
                        ).props("outline color=secondary icon=lock_reset")

        def update_admin_status(target_user_id: int, is_admin: bool) -> None:
            """Update admin privileges for a specific user."""
            current_user_id = app.storage.user.get("user_id")
            db = SessionLocal()
            try:
                target_user = (
                    db.query(User).filter(User.id == target_user_id).first()
                )
                if not target_user:
                    ui.notify("User not found.", type="negative")
                    refresh_user_management()
                    return

                if target_user.id == current_user_id and not is_admin:
                    # SQLAlchemy requires == True for Boolean column
                    # expressions
                    admin_count = db.query(User).filter(
                        User.is_admin == True  # noqa: E712
                    ).count()
                    if admin_count <= 1:
                        ui.notify(
                            "You cannot remove admin rights from the "
                            "last account.",
                            type="warning"
                        )
                        refresh_user_management()
                        return

                target_user.is_admin = is_admin
                db.commit()

                if target_user.id == current_user_id:
                    app.storage.user["is_admin"] = bool(is_admin)

                ui.notify(
                    f"Updated status for '{target_user.username}'.",
                    type="positive"
                )
            finally:
                db.close()

            refresh_user_management()

        def reset_password(target_user_id: int) -> None:
            """Change the password for a user."""
            input_field = password_inputs.get(target_user_id)
            new_password = (input_field.value if input_field else "") or ""
            new_password = new_password.strip()

            if len(new_password) < 4:
                ui.notify("Password too short.", type="warning")
                return

            db = SessionLocal()
            try:
                target_user = (
                    db.query(User).filter(User.id == target_user_id).first()
                )
                if not target_user:
                    ui.notify("User not found.", type="negative")
                    refresh_user_management()
                    return

                target_user.password_hash = hash_password(new_password)
                db.commit()

                if input_field:
                    input_field.value = ""

                ui.notify(
                    f"Password reset for '{target_user.username}'.",
                    type="positive"
                )
            finally:
                db.close()

        def create_new_user() -> None:
            """Register a new user via the admin form."""
            username = (username_input.value or "").strip()
            password = (password_input.value or "").strip()
            is_admin = bool(admin_toggle.value)

            if not username or not password:
                ui.notify("Username and password required.", type="warning")
                return

            if get_user_by_username(username):
                ui.notify(f"User '{username}' exists.", type="warning")
                return

            create_user(username, password, is_admin=is_admin)
            username_input.value = ""
            password_input.value = ""
            admin_toggle.value = False
            ui.notify(f"User '{username}' created.", type="positive")
            refresh_user_management()

        with professional_layout("Admin Dashboard"):
            ui.label("Administrative Control Panel").classes(
                "text-lg text-slate-600 mb-4"
            )
            with ui.row().classes("gap-6"):
                ui.button(
                    "New drawing", on_click=lambda: ui.navigate.to("/")
                ).props("color=primary icon=brush")
                ui.button(
                    "View all user submissions",
                    on_click=lambda: ui.navigate.to("/admin/history")
                ).props("outline icon=manage_search")

            with ui.card().classes(
                "w-full mt-6 p-4 border border-slate-200 shadow-sm"
            ):
                ui.label("Create New User").classes(
                    "text-lg font-semibold text-slate-700 mb-3"
                )
                with ui.column().classes("w-full gap-3"):
                    username_input = ui.input("Username").props(
                        "outlined dense"
                    ).classes("w-full")
                    password_input = ui.input(
                        "Password", password=True, password_toggle_button=True
                    ).props("outlined dense").classes("w-full")
                    admin_toggle = ui.checkbox("Grant admin access")
                    ui.button(
                        "New User", on_click=create_new_user
                    ).props("color=primary icon=person_add")

            with ui.card().classes(
                "w-full mt-6 p-4 border border-slate-200 shadow-sm"
            ):
                ui.label("Manage User Accounts").classes(
                    "text-lg font-semibold text-slate-700 mb-3"
                )
                user_management_container = ui.column().classes("w-full gap-2")

            refresh_user_management()


class AdminHistoryPage:
    """View to display and manage all global prediction entries."""

    def render(self):
        """Render the submission history table."""
        if not _is_current_user_admin():
            ui.notify("Admin access only.", type="negative")
            ui.navigate.to("/")
            return

        def load_all_entries():
            """Fetch prediction entries with associated usernames."""
            db = SessionLocal()
            try:
                entries = (
                    db.query(PredictionEntry, User.username)
                    .join(User, PredictionEntry.user_id == User.id)
                    .order_by(PredictionEntry.created_at.desc())
                    .all()
                )
                rows = []
                for entry, username in entries:
                    original_base64 = base64.b64encode(
                        entry.original_image
                    ).decode("utf-8")
                    rows.append({
                        "id": entry.id,
                        "username": username,
                        "prediction": entry.prediction,
                        "model_name": entry.model_name,
                        "probability": entry.probability,
                        "original": f"data:image/png;base64,{original_base64}",
                        "date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                return rows
            finally:
                db.close()

        async def admin_delete_entry(entry_id):
            """Delete a prediction entry permanently."""
            db = SessionLocal()
            try:
                entry = db.query(PredictionEntry).filter(
                    PredictionEntry.id == entry_id
                ).first()
                if entry:
                    db.delete(entry)
                    db.commit()
                    ui.notify(f"Admin: Entry {entry_id} deleted.")
                    table.rows[:] = load_all_entries()
            except Exception as e:
                ui.notify(f"Error: {e}", type='negative')
            finally:
                db.close()

        with professional_layout("Global Submission History"):
            rows = load_all_entries()
            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True},
                {"name": "username", "label": "User", "field": "username"},
                {"name": "original", "label": "Drawing", "field": "original"},
                {"name": "prediction", "label": "Prediction", "field": "pred"},
                {"name": "model_name", "label": "Model", "field": "model"},
                {"name": "probability", "label": "Conf", "field": "prob"},
                {"name": "date", "label": "Timestamp", "field": "date"},
                {"name": "delete", "label": "Actions", "field": "id"}
            ]

            table = ui.table(
                columns=columns, rows=rows, row_key="id"
            ).classes("w-full shadow-lg border")

            table.add_slot("body-cell-original", r'''
            <q-td :props="props">
              <img :src="props.value" style="width:60px;height:60px;" />
            </q-td>
            ''')

            table.add_slot("body-cell-delete", r'''
            <q-td :props="props">
              <q-btn flat round icon="delete" color="red"
                @click="$parent.$emit('admin_delete', props.value)" />
            </q-td>
            ''')

            table.on(
                'admin_delete', lambda msg: admin_delete_entry(msg.args)
            )

            if not rows:
                ui.label("No entries found.").classes("mt-10 text-gray-400")
