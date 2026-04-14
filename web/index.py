import io
import datetime
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from nicegui import ui, app
import PIL.ImageOps as ImageOps
from DB.database import SessionLocal, PredictionEntry
from ml.registry import get_recognizer, AVAILABLE_MODELS
from web.layout import professional_layout

class LandingPage:
    def __init__(self):
        self.title = "Handwritten Digit Recognizer"
        self.path = []
        self.ii = None
        self.selected_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "cnn"

    def render(self):
        with professional_layout(self.title):
            ui.label('Draw a digit (0-9) below:').classes('text-xl text-slate-600')

            self.ii = ui.interactive_image(
                size=(500, 500),
                on_mouse=self.handle_mouse,
                events=['mousedown', 'mousemove', 'mouseup'],
                cross=False
            ).classes(
                'border-4 border-slate-300 rounded-xl bg-white cursor-crosshair shadow-inner hover:border-blue-400 transition-all'
            ).style('width: 500px; height: 500px;')

            with ui.column().classes("w-full items-center gap-4 mt-4"):
                ui.label("Select AI Model").classes("text-xl font-bold text-slate-800")
                ui.select(
                    options=AVAILABLE_MODELS, 
                    on_change=lambda e: ui.notify(f"Selected: {e.value}")
                ).classes("w-64").bind_value(self, 'selected_model').props('outlined dense')

                with ui.row().classes('mt-4 space-x-4'):
                    ui.button('Clear', on_click=self.clear_canvas).props('outline color=negative icon=delete')
                    ui.button('Predict & Save', on_click=self.process_drawing).props('color=primary icon=auto_awesome shadow')

    def handle_mouse(self, e):
        if e.type == 'mousedown':
            self.path = [(e.image_x, e.image_y)]
        elif e.type == 'mousemove' and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))
            svg_path = ' '.join([f'{"M" if i == 0 else "L"} {p[0]} {p[1]}' for i, p in enumerate(self.path)])
            new_stroke = f'''
            <path d="{svg_path}" stroke="#1e293b" stroke-width="20" fill="none" stroke-linecap="round" stroke-linejoin="round" />
            '''
            self.ii.content += new_stroke

    def clear_canvas(self):
        self.path = []
        self.ii.content = ""

    async def process_drawing(self):
        if not self.ii.content:
            ui.notify("Please draw something first!", type='warning')
            return
            
        current_user_id = app.storage.user.get('user_id')
        if not current_user_id:
            ui.notify("Error: No user session. Please log in.", type='negative')
            return

        try:
    # 1. Convert SVG to high-res PNG bytes
            full_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">{self.ii.content}</svg>'
            svg_file = io.BytesIO(full_svg.encode('utf-8'))
            drawing = svg2rlg(svg_file)
            original_png_bytes = renderPM.drawToString(drawing, fmt="PNG")

            # 2. Open image and convert to Grayscale ('L')
            # This is necessary because MNIST models expect single-channel inputs
            img = Image.open(io.BytesIO(original_png_bytes)).convert('L')

            # 3. Resize to 28x28 (LANCZOS is good for preserving digit structure)
            img_small = img.resize((28, 28), Image.Resampling.LANCZOS)

            # 4. Invert the small image (from Black-on-White to White-on-Black)
            # This aligns the image with the MNIST training data format
            img_inverted = ImageOps.invert(img_small)
            
            # 5. Save the processed image back into bytes
            small_buffer = io.BytesIO()
            img_inverted.save(small_buffer, format="PNG")
            downsized_png_bytes = small_buffer.getvalue()

            recognizer = get_recognizer(self.selected_model)
            result = recognizer.predict_from_png_bytes(original_png_bytes)
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

            ui.notify(f'Prediction: {predicted_digit}', type='positive')
            self.clear_canvas()
        except Exception as e:
            ui.notify(f'Error: {str(e)}', type='negative')