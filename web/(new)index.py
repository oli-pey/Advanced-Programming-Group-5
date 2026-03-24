import io
import datetime

from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from nicegui import ui, app

from DB.database import SessionLocal, PredictionEntry
from ml.predictor import predict_digit


class LandingPage:
    def __init__(self):
        self.title = "Handwritten Digit Recognizer"
        self.path = []
        self.ii = None

    def render(self):
        username = app.storage.user.get("username")
        is_admin = app.storage.user.get("is_admin", False)

        with ui.header().classes("bg-primary text-white p-4 justify-between items-center"):
            ui.label(self.title).classes("text-2xl font-bold")

            with ui.row().classes("items-center gap-2"):
                ui.label(f"Logged in as: {username}")
                ui.button("My History", on_click=lambda: ui.navigate.to("/history")).props(
                    "flat color=white icon=history"
                )

                if is_admin:
                    ui.button("Admin", on_click=lambda: ui.navigate.to("/admin")).props(
                        "flat color=white icon=admin_panel_settings"
                    )

                ui.button("Logout", on_click=self.logout).props(
                    "flat color=white icon=logout"
                )

        with ui.column().classes("w-full items-center mt-10 space-y-4"):
            ui.label("Draw a digit (0-9) below:").classes("text-xl")

            self.ii = ui.interactive_image(
                size=(500, 500),
                on_mouse=self.handle_mouse,
                events=["mousedown", "mousemove", "mouseup"],
                cross=False,
            ).classes(
                "border-4 border-gray-400 bg-white cursor-crosshair shadow-lg"
            ).style("width: 500px; height: 500px;")

            with ui.row().classes("mt-4 space-x-4"):
                ui.button("Clear", on_click=self.clear_canvas).props("outline color=red")
                ui.button("Predict & Save", on_click=self.process_drawing).props("color=primary")

    def logout(self):
        app.storage.user.clear()
        ui.navigate.to("/login")

    def handle_mouse(self, e):
        if e.type == "mousedown":
            self.path = [(e.image_x, e.image_y)]

        elif e.type == "mousemove" and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))
            svg_path = " ".join(
                [f'{"M" if i == 0 else "L"} {p[0]} {p[1]}' for i, p in enumerate(self.path)]
            )
            new_stroke = f'''
            <path d="{svg_path}"
                  stroke="black"
                  stroke-width="18"
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
            ui.notify("Please draw something first!", type="warning")
            return

        user_id = app.storage.user.get("user_id")
        if not user_id:
            ui.notify("Please log in first.", type="negative")
            ui.navigate.to("/login")
            return

        try:
            full_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">{self.ii.content}</svg>'
            svg_file = io.BytesIO(full_svg.encode("utf-8"))

            drawing = svg2rlg(svg_file)
            original_png_bytes = renderPM.drawToString(drawing, fmt="PNG")

            img = Image.open(io.BytesIO(original_png_bytes)).convert("L")
            img_small = img.resize((28, 28), Image.Resampling.LANCZOS)

            small_buffer = io.BytesIO()
            img_small.save(small_buffer, format="PNG")
            downsized_png_bytes = small_buffer.getvalue()

            # === NEW: ML call from ml/ folder ===
            prediction = predict_digit(downsized_png_bytes)

            db = SessionLocal()
            try:
                new_entry = PredictionEntry(
                    user_id=user_id,
                    original_image=original_png_bytes,
                    downsized_image=downsized_png_bytes,
                    prediction=str(prediction),
                    created_at=datetime.datetime.utcnow(),
                )
                db.add(new_entry)
                db.commit()
            finally:
                db.close()

            ui.notify(f"Prediction: {prediction}", type="positive")
            self.clear_canvas()

        except Exception as e:
            ui.notify(f"Processing Error: {str(e)}", type="negative")
