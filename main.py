import io
import datetime

from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from nicegui import ui, app
from DB.database import SessionLocal, PredictionEntry
from ml import result
from ml.registry import get_recognizer, AVAILABLE_MODELS
from web.index import LandingPage
from web.layout import professional_layout
from web.history import HistoryPage
from web.admin import AdminDashboard, AdminHistoryPage
from auth import bootstrap_defaults

bootstrap_defaults()


class LandingPage:
    def __init__(self):
        self.title = "Draw a Digit"
        self.path = []
        self.ii = None
        self.selected_model = (
            AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "cnn"
        )

    def render(self):
        with professional_layout(self.title):

            ui.label(
                "Use your mouse or touch to draw a single digit (0-9)"
            ).classes("text-lg text-slate-600")

            self.ii = ui.interactive_image(
                size=(500, 500),
                on_mouse=self.handle_mouse,
                events=['mousedown', 'mousemove', 'mouseup'],
                cross=False
            ).classes(
                'border-4 border-slate-300 rounded-xl bg-white '
                'cursor-crosshair shadow-inner hover:border-blue-400'
            ).style('width: 500px; height: 500px;')

            with ui.row().classes(
                "w-full items-center justify-between gap-6 mt-4 "
                "p-4 bg-slate-50 rounded-xl border border-slate-200"
            ):

                with ui.column().classes("gap-1"):
                    ui.label("Select AI Model").classes(
                        "text-xs font-bold text-slate-500 uppercase"
                    )

                    ui.select(
                        options=AVAILABLE_MODELS,
                        on_change=lambda e: ui.notify(
                            f"Model ready: {e.value}"
                        )
                    ).classes("w-48").bind_value(
                        self, 'selected_model'
                    ).props('outlined dense')

                with ui.row().classes('gap-4'):
                    ui.button(
                        'Clear Canvas',
                        on_click=self.clear_canvas
                    ).props('outline color=negative icon=delete')

                    ui.button(
                        'Predict & Save',
                        on_click=self.process_drawing
                    ).props('color=primary icon=auto_awesome')

    def handle_mouse(self, e):
        if e.type == 'mousedown':
            self.path = [(e.image_x, e.image_y)]

        elif e.type == 'mousemove' and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))

            svg_path = ' '.join([
                f'{"M" if i == 0 else "L"} {p[0]} {p[1]}'
                for i, p in enumerate(self.path)
            ])

            new_stroke = f'''
            <path d="{svg_path}"
                  stroke="#1e293b"
                  stroke-width="20"
                  fill="none"
                  stroke-linecap="round"
                  stroke-linejoin="round" />
            '''

            self.ii.content += new_stroke

    def clear_canvas(self):
        self.path = []
        self.ii.content = ""

    async def process_drawing(self):
        if not self.ii.content:
            ui.notify(
                "Please draw something first!",
                type='warning'
            )
            return

        current_user_id = app.storage.user.get('user_id')

        if not current_user_id:
            ui.notify(
                "No user session found. Please log in again.",
                type='negative'
            )
            return

        try:
            full_svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="500" height="500">{self.ii.content}</svg>'
            )

            svg_file = io.BytesIO(full_svg.encode('utf-8'))

            drawing = svg2rlg(svg_file)
            original_png_bytes = renderPM.drawToString(
                drawing,
                fmt="PNG"
            )

            img = Image.open(
                io.BytesIO(original_png_bytes)
            ).convert('L')

            img_small = img.resize(
                (28, 28),
                Image.Resampling.LANCZOS
            )

            small_buffer = io.BytesIO()
            img_small.save(small_buffer, format="PNG")
            downsized_png_bytes = small_buffer.getvalue()

            recognizer = get_recognizer(self.selected_model)

            result = recognizer.predict_from_png_bytes(
                original_png_bytes
            )

            predicted_digit = str(result.predicted_digit)
            model_used = result.model_name
            confidence_score = str(result.confidence)

            db = SessionLocal()

            try:
                new_entry = PredictionEntry(
                    user_id=current_user_id,
                    original_image=original_png_bytes,
                    downsized_image=downsized_png_bytes,
                    prediction=predicted_digit,
                    model_name=model_used,        # Save algorithm name
                    probability=confidence_score,
                    created_at=datetime.datetime.utcnow()
                )

                db.add(new_entry)
                db.commit()

            finally:
                db.close()

            ui.notify(
                f'Prediction ({self.selected_model.upper()}): '
                f'{predicted_digit}',
                type='positive'
            )

            self.clear_canvas()

        except Exception as e:
            ui.notify(
                f'Processing Error: {str(e)}',
                type='negative'
            )
@ui.page('/login')
def login_page():
    # We can use your professional layout, or just standard UI
    with professional_layout("Login"):
        with ui.card().classes('mx-auto mt-20 p-8 w-96 items-center shadow-lg'):
            ui.label('Welcome Back').classes('text-2xl font-bold mb-4 text-slate-800')
            
            # Input fields
            username = ui.input('Username').classes('w-full mb-2').props('outlined')
            password = ui.input('Password', password=True, password_toggle_button=True).classes('w-full mb-6').props('outlined')
            
            # Login action
            def try_login():
                # 1. Check database using your auth.py logic
                user = auth.authenticate_user(username.value, password.value)
                
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

                        if auth.get_user_by_username(username_value):
                            ui.notify('Username already exists.', type='warning')
                            return

                        user = auth.create_user(username_value, password_value, is_admin=False)
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
            ui.button('Create New Account', on_click=open_signup_dialog).classes('w-full mt-2').props('outline icon=person_add')

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
    if not require_session():
        return
    AdminDashboard().render()

@ui.page('/admin/history')
def admin_history_page():
    if not require_session():
        return
    AdminHistoryPage().render()

if __name__ in {"__main__", "__mp_main__"}:
    import auth
    auth.bootstrap_defaults() # This ensures 'admin' exists with admin rights
    ui.run(storage_secret='PICK_A_SECURE_PASSWORD')