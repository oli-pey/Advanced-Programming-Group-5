from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from nicegui import ui, app
from web.index import LandingPage
from web.history import HistoryPage
from web.admin import AdminDashboard, AdminHistoryPage
from auth import authenticate_user, bootstrap_defaults
from auth import create_user, get_user_by_username

bootstrap_defaults()

landing = LandingPage()
history = HistoryPage()
admin_dashboard = AdminDashboard()
admin_history = AdminHistoryPage()

UNRESTRICTED_ROUTES = {"/login","/register"}

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

@ui.page("/register")
def register_page():
    if app.storage.user.get("authenticated", False):
        return RedirectResponse("/")

    def try_register():
        user_val = username.value.strip()
        pass_val = password.value
        confirm_val = confirm_password.value

        if not user_val or not pass_val:
            ui.notify("Username and password are required", color="negative")
            return
        
        if pass_val != confirm_val:
            ui.notify("Passwords do not match", color="negative")
            return

        if get_user_by_username(user_val):
            ui.notify("Username already exists", color="negative")
            return

        # Create the user in the database
        create_user(user_val, pass_val, is_admin=False)
        ui.notify("Account created successfully! Please log in.", color="positive")
        ui.navigate.to("/login")

    with ui.card().classes("absolute-center w-96 p-6"):
        ui.label("Create Account").classes("text-2xl font-bold")
        username = ui.input("Username").classes("w-full")
        password = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")
        confirm_password = ui.input("Confirm Password", password=True, password_toggle_button=True).classes("w-full")
        
        with ui.row().classes("w-full justify-between items-center mt-4"):
            ui.button("Sign Up", on_click=try_register).props("color=primary")
            ui.link("Back to Login", "/login").classes("text-sm text-blue-500")

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
        with ui.card().classes("absolute-center w-96 p-6"):
            ui.label("Login").classes("text-2xl font-bold")

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
    with ui.row().classes("w-full justify-between items-center mt-4"):
            ui.button("Log in", on_click=try_login).props("color=primary")
            ui.link("Create Account", "/register").classes("text-sm text-blue-500") # Add this link


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
