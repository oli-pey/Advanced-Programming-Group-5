from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from nicegui import ui, app

from web.index import LandingPage
from web.history import HistoryPage
from web.admin import AdminDashboard, AdminHistoryPage
from auth import authenticate_user, bootstrap_defaults

bootstrap_defaults()

landing = LandingPage()
history = HistoryPage()
admin_dashboard = AdminDashboard()
admin_history = AdminHistoryPage()

UNRESTRICTED_ROUTES = {"/login"}

ADMIN_ONLY_ROUTES = {"/admin", "/admin/history"}


@app.add_middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 1. ALLOW internal NiceGUI traffic and static files
        # Without this, the login page can't "talk" to the server, causing the loop
        if path.startswith("/_nicegui") or "." in path.split("/")[-1]:
            return await call_next(request)

        is_logged_in = app.storage.user.get("authenticated", False)

        # 2. REDIRECT to login if not authenticated
        if not is_logged_in and path not in UNRESTRICTED_ROUTES:
            return RedirectResponse("/login")

        # 3. PREVENT authenticated users from getting stuck on the login page
        if is_logged_in and path == "/login":
            return RedirectResponse("/")

        return await call_next(request)


@ui.page("/login")
def login_page():
    if app.storage.user.get("authenticated", False):
        return RedirectResponse("/")

    def try_login():
        user = authenticate_user(username.value.strip(), password.value)
        if not user:
            ui.notify("Wrong username or password", color="negative")
            return

        app.storage.user["authenticated"] = True
        app.storage.user["user_id"] = user.id
        app.storage.user["username"] = user.username
        app.storage.user["is_admin"] = user.is_admin

        ui.navigate.to("/")

    with ui.card().classes("absolute-center w-96 p-6"):
        ui.label("Login").classes("text-2xl font-bold")
        username = ui.input("Username").classes("w-full").on("keydown.enter", try_login)
        password = ui.input(
            "Password",
            password=True,
            password_toggle_button=True,
        ).classes("w-full").on("keydown.enter", try_login)
        ui.button("Log in", on_click=try_login).props("color=primary")


@ui.page("/")
def main_page():
    landing.render()


@ui.page("/history")
def history_view():
    history.render()


@ui.page("/admin")
def admin_page():
    admin_dashboard.render()


@ui.page("/admin/history")
def admin_history_page():
    admin_history.render()

@ui.page('/logout')
def logout_page():
    app.storage.user.clear()  # This removes the "authenticated" key
    return RedirectResponse('/login')


ui.run(
    title="Advanced Programming Project - Group 5",
    port=8080,
    storage_secret="CHANGE_THIS_TO_A_LONG_RANDOM_SECRET",
)
