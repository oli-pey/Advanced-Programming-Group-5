from contextlib import contextmanager

from nicegui import app, ui


def logout_user():
    """Navigate the user to the logout route."""
    ui.navigate.to('/logout')


@contextmanager
def professional_layout(page_title: str):
    """
    A context manager that provides a consistent, professional-grade
    UI shell for the application. Includes a responsive header with
    navigation, custom color themes, and a centered content card.

    Args:
        page_title (str): The title displayed at the top of the main card.
    """
    # Retrieve authentication state and permissions from storage
    is_authenticated = bool(app.storage.user.get('user_id'))
    is_admin = bool(app.storage.user.get('is_admin', False))

    # Configure application theme colors
    ui.colors(
        primary='#2563eb',
        secondary='#64748b',
        accent='#0ea5e9',
        positive='#16a34a',
        negative='#dc2626'
    )

    # Set global body styling for a modern look
    ui.query('body').style('''
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        margin: 0;
        font-family: Inter, sans-serif;
    ''')

    # Application Header
    with ui.header().classes(
        'bg-white shadow-md px-6 py-4 flex justify-between items-center'
    ):
        with ui.row().classes('items-center gap-3'):
            ui.icon('psychology', size='lg').classes('text-blue-600')
            ui.label('AI Digit Recognizer').classes(
                'text-xl font-bold text-slate-800'
            )

        # Dynamic Navigation based on user status
        if is_authenticated:
            with ui.row().classes('gap-2'):
                ui.button(
                    'Home',
                    on_click=lambda: ui.navigate.to('/')
                ).props('flat icon=home')

                ui.button(
                    'Sandbox',
                    on_click=lambda: ui.navigate.to('/sandbox')
                ).props('flat icon=science')

                ui.button(
                    'History',
                    on_click=lambda: ui.navigate.to('/history')
                ).props('flat icon=history')

                if is_admin:
                    ui.button(
                        'Admin',
                        on_click=lambda: ui.navigate.to('/admin')
                    ).props('flat icon=admin_panel_settings')

                ui.button(
                    'Logout',
                    on_click=logout_user
                ).props('flat color=negative icon=logout')

    # Main Content Area
    with ui.column().classes(
        'w-full min-h-screen items-center p-8'
    ):
        with ui.card().classes(
            'w-full max-w-5xl p-8 rounded-2xl shadow-xl bg-white'
        ):
            ui.label(page_title).classes(
                'text-3xl font-bold text-slate-800 mb-6'
            )

            # Inject the specific page content here
            yield
