import io
import datetime

from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from nicegui import ui, app

from DB.database import SessionLocal, PredictionEntry
from ml.registry import get_recognizer
from ml.registry import get_recognizer

try:
    recognizer = get_recognizer("cnn")
except Exception as e:
    recognizer = None
    print(f"Recognizer failed to load: {e}")


class LandingPage:
    def __init__(self):
        self.title = "Handwritten Digit Recognizer"
        self.path = []
        self.ii = None
        self.selected_model = list(get_recognizer.keys())[0] if get_recognizer.keys() else "CNN"

    # web/index.py

    def render(self):
        with ui.header().classes('bg-primary text-white p-4 justify-between items-center'):
            ui.label(self.title).classes('text-2xl font-bold')
            # Inside web/index.py -> LandingPage.render()
            with ui.row().classes('items-center space-x-2'):
                ui.button('History', on_click=lambda: ui.navigate.to('/history')).props('flat color=white icon=history')
                # Use /logout instead of /login to ensure the session is cleared
                ui.button('Logout', on_click=lambda: ui.navigate.to('/logout')).props('flat color=white icon=logout')


        with ui.column().classes('w-full items-center mt-10 space-y-4'):
            ui.label('Draw a digit (0-9) below:').classes('text-xl')

            self.ii = ui.interactive_image(
                size=(500, 500),
                on_mouse=self.handle_mouse,
                events=['mousedown', 'mousemove', 'mouseup'],
                cross=False
            ).classes(
                'border-4 border-gray-400 bg-white cursor-crosshair shadow-lg'
            ).style('width: 500px; height: 500px;')

        with ui.column().classes("w-full items-center gap-4 mt-10"):
            ui.label("Select AI Model").classes("text-xl font-bold")
            ui.select(
                options=list(get_recognizer().keys()), 
                label="Choose Algorithm",
                on_change=lambda e: ui.notify(f"Selected: {e.value}")
            ).classes("w-64").bind_value(self, 'selected_model')

            with ui.row().classes('mt-4 space-x-4'):
                ui.button('Clear', on_click=self.clear_canvas).props('outline color=red')
                ui.button('Predict & Save', on_click=self.process_drawing).props('color=primary')

    def handle_mouse(self, e):
        if e.type == 'mousedown':
            self.path = [(e.image_x, e.image_y)]

        elif e.type == 'mousemove' and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))
            svg_path = ' '.join([f'{"M" if i == 0 else "L"} {p[0]} {p[1]}' for i, p in enumerate(self.path)])
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

    async def handle_prediction(self, image_data):
        # 2. Use the selected model for prediction
        algorithm_name = self.selected_model
        model_instance = get_recognizer().get(algorithm_name)
        if model_instance:
            # result = model_instance.predict(image_data)
            ui.notify(f"Predicting with {algorithm_name}...")

    async def process_drawing(self):
        """Application Logic: convert drawing, run ML, save prediction"""
        if not self.ii.content:
            ui.notify("Please draw something first!", type='warning')
            return
        current_user_id = app.storage.user.get('user_id')
        if not current_user_id:
            ui.notify("Error: No user session found. Please log in again.", type='negative')
            return

        try:
            full_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">{self.ii.content}</svg>'
            svg_file = io.BytesIO(full_svg.encode('utf-8'))

            drawing = svg2rlg(svg_file)
            original_png_bytes = renderPM.drawToString(drawing, fmt="PNG")

            img = Image.open(io.BytesIO(original_png_bytes)).convert('L')
            img_small = img.resize((28, 28), Image.Resampling.LANCZOS)

            small_buffer = io.BytesIO()
            img_small.save(small_buffer, format="PNG")
            downsized_png_bytes = small_buffer.getvalue()

            if recognizer is None:
                ui.notify("ML model not available", type='negative')
                return
            result = recognizer.predict_from_png_bytes(original_png_bytes)
            predicted_digit = str(result.predicted_digit)

            db = SessionLocal()
            try:
                new_entry = PredictionEntry(
                    user_id=app.storage.user.get('user_id'),  # Add this line
                    original_image=original_png_bytes,
                    downsized_image=downsized_png_bytes,
                    prediction=predicted_digit, 
                    created_at=datetime.datetime.utcnow()
                )
                db.add(new_entry)
                db.commit()
            finally:
                db.close()

            ui.notify(f'Prediction: {predicted_digit}', type='positive')
            self.clear_canvas()

        except Exception as e:
            ui.notify(f'Processing Error: {str(e)}', type='negative')
