from nicegui import ui, app

from auth import authenticate_user, create_user, get_user_by_username, require_session
from web.index import LandingPage
from web.history import HistoryPage
from web.layout import professional_layout

# import these only if they really exist
# from web.admin import AdminDashboard
# from web.admin_history import AdminHistoryPage


def register_routes() -> None:

    @ui.page('/login')
    def login_page():
        with professional_layout("Login"):
            with ui.card().classes('mx-auto mt-20 p-8 w-96 items-center shadow-lg'):
                ui.label('Welcome Back').classes('text-2xl font-bold mb-4 text-slate-800')

                username = ui.input('Username').classes('w-full mb-2').props('outlined')
                password = ui.input(
                    'Password',
                    password=True,
                    password_toggle_button=True
                ).classes('w-full mb-6').props('outlined')

                def try_login():
                    user = authenticate_user(username.value, password.value)

                    if user:
                        app.storage.user['user_id'] = user.id
                        app.storage.user['username'] = user.username
                        app.storage.user['is_admin'] = bool(user.is_admin)

                        ui.notify('Logged in successfully!', type='positive')
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
                            password_toggle_button=True
                        ).classes('w-full').props('outlined')
                        confirm_password = ui.input(
                            'Confirm Password',
                            password=True,
                            password_toggle_button=True
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
                            ui.button('Create Account', on_click=create_account).props(
                                'color=primary icon=person_add'
                            )

                    dialog.open()

                ui.button('Log In', on_click=try_login).classes('w-full').props('color=primary size=lg')
                ui.button('Create New Account', on_click=open_signup_dialog).classes('w-full mt-2').props(
                    'outline icon=person_add'
                )

    @ui.page('/logout')
    def logout_page():
        app.storage.user.clear()
        ui.notify('You have been logged out.', type='info')
        ui.navigate.to('/login')

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

    # only enable these when the classes exist
    # @ui.page('/admin')
    # def admin_page():
    #     if not require_session():
    #         return
    #     AdminDashboard().render()

    # @ui.page('/admin/history')
    # def admin_history_page():
    #     if not require_session():
    #         return
    #     AdminHistoryPage().render()