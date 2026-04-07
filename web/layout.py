from nicegui import ui, app
from contextlib import contextmanager

def logout_user():
    ui.navigate.to('/logout')

@contextmanager
def professional_layout(page_title: str):
    is_authenticated = bool(app.storage.user.get('user_id'))
    is_admin = bool(app.storage.user.get('is_admin', False))

    ui.colors(
        primary='#2563eb',
        secondary='#64748b',
        accent='#0ea5e9',
        positive='#16a34a',
        negative='#dc2626'
    )

    ui.query('body').style('''
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        margin: 0;
        font-family: Inter, sans-serif;
    ''')

    with ui.header().classes(
        'bg-white shadow-md px-6 py-4 flex justify-between items-center'
    ):
        with ui.row().classes('items-center gap-3'):
            ui.icon('psychology', size='lg').classes('text-blue-600')
            ui.label('AI Digit Recognizer').classes(
                'text-xl font-bold text-slate-800'
            )

        if is_authenticated:
            with ui.row().classes('gap-2'):
                ui.button('Home', on_click=lambda: ui.navigate.to('/')).props('flat icon=home')
                ui.button('History', on_click=lambda: ui.navigate.to('/history')).props('flat icon=history')
                if is_admin:
                    ui.button('Admin', on_click=lambda: ui.navigate.to('/admin')).props('flat icon=admin_panel_settings')
                ui.button('Logout', on_click=logout_user).props('flat color=negative icon=logout')

    with ui.column().classes(
        'w-full min-h-screen items-center p-8'
    ):
        with ui.card().classes(
            'w-full max-w-5xl p-8 rounded-2xl shadow-xl bg-white'
        ):
            ui.label(page_title).classes(
                'text-3xl font-bold text-slate-800 mb-6'
            )

            yield