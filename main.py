import io
import datetime

from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from nicegui import ui, app

from DB.database import SessionLocal, PredictionEntry
from ml.registry import get_recognizer, AVAILABLE_MODELS
from web.layout import professional_layout
from web.history import HistoryPage


class AdminDashboard:
    def render(self):
        with professional_layout("Admin Dashboard"):
            ui.label("Admin dashboard is coming soon").classes(
                "text-lg text-slate-600"
            )


class AdminHistoryPage:
    def render(self):
        with professional_layout("Admin History"):
            ui.label("Admin history page is coming soon").classes(
                "text-lg text-slate-600"
            )


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

            db = SessionLocal()

            try:
                new_entry = PredictionEntry(
                    user_id=current_user_id,
                    original_image=original_png_bytes,
                    downsized_image=downsized_png_bytes,
                    prediction=predicted_digit,
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


# ROUTES
ui.route("/", LandingPage().render)
ui.route("/history", HistoryPage().render)
ui.route("/admin", AdminDashboard().render)
ui.route("/admin/history", AdminHistoryPage().render)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(storage_secret='PICK_A_SECURE_PASSWORD')