import io
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps
from nicegui import app, ui
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

from DB.database import PredictionEntry, SessionLocal
from ml.registry import AVAILABLE_MODELS, get_recognizer
from web.layout import professional_layout


class LandingPage:
    """
    The main application page where users can draw digits on a canvas,
    select an AI model, and get real-time predictions.
    """

    def __init__(self):
        self.title = "Draw a Digit"
        self.path = []
        self.ii = None
        self.selected_model = (
            AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "cnn"
        )

    def render(self):
        """Render the landing page UI components."""
        with professional_layout(self.title):

            ui.label(
                "Use your mouse or touch to draw a single digit (0-9)"
            ).classes("text-lg text-slate-600")

            # Interactive drawing canvas
            self.ii = ui.interactive_image(
                size=(500, 500),
                on_mouse=self.handle_mouse,
                events=['mousedown', 'mousemove', 'mouseup'],
                cross=False
            ).classes(
                'border-4 border-slate-300 rounded-xl bg-white '
                'cursor-crosshair shadow-inner hover:border-blue-400'
            ).style('width: 500px; height: 500px;')

            # Controls Bar
            with ui.row().classes(
                "w-full items-center justify-between gap-6 mt-4 "
                "p-4 bg-slate-50 rounded-xl border border-slate-200"
            ):

                # Model Selection
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

                # Action Buttons
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
        """
        Manage drawing logic. Converts mouse movements into 
        SVG path elements displayed on the interactive image.
        """
        if e.type == 'mousedown':
            self.path = [(e.image_x, e.image_y)]

        elif e.type == 'mousemove' and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))

            # Generate SVG string from points
            svg_path = ' '.join([
                f'{"M" if i == 0 else "L"} {p[0]} {p[1]}'
                for i, p in enumerate(self.path)
            ])

            # Append the new stroke to the canvas content
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
        """Reset the drawing path and clear the SVG content."""
        self.path = []
        self.ii.content = ""

    async def process_drawing(self):
        """
        Pre-process the drawing (resize/invert), run ML inference, 
        and save results to the database.
        """
        if not self.ii.content:
            ui.notify("Please draw something first!", type='warning')
            return

        current_user_id = app.storage.user.get('user_id')
        if not current_user_id:
            ui.notify("Session expired. Please log in.", type='negative')
            return

        try:
            # 1. SVG to PNG conversion
            svg_header = '<svg xmlns="http://www.w3.org/2000/svg" '
            svg_size = 'width="500" height="500">'
            full_svg = f'{svg_header}{svg_size}{self.ii.content}</svg>'
            
            svg_file = io.BytesIO(full_svg.encode('utf-8'))
            drawing = svg2rlg(svg_file)
            original_png_bytes = renderPM.drawToString(drawing, fmt="PNG")

            # 2. Image Processing for ML (MNIST-style)
            img = Image.open(io.BytesIO(original_png_bytes)).convert('L')
            
            # Resize to 28x28 (LANCZOS preserves structure)
            img_small = img.resize((28, 28), Image.Resampling.LANCZOS)

            # Invert colors (MNIST is White-on-Black)
            img_inverted = ImageOps.invert(img_small)

            # 3. Buffer management for downsized image
            small_buffer = io.BytesIO()
            img_inverted.save(small_buffer, format="PNG")
            downsized_png_bytes = small_buffer.getvalue()

            # 4. Machine Learning Inference
            recognizer = get_recognizer(self.selected_model)
            result = recognizer.predict_from_png_bytes(original_png_bytes)

            # 5. Database Persistence
            db = SessionLocal()
            try:
                new_entry = PredictionEntry(
                    user_id=current_user_id,
                    original_image=original_png_bytes,
                    downsized_image=downsized_png_bytes,
                    prediction=str(result.predicted_digit),
                    model_name=result.model_name,
                    probability=str(result.confidence),
                    created_at=datetime.now(ZoneInfo('Europe/Prague'))
                )
                db.add(new_entry)
                db.commit()
            finally:
                db.close()

            # 6. UI Feedback
            ui.notify(
                f'Prediction ({result.model_name.upper()}): '
                f'{result.predicted_digit}',
                type='positive'
            )
            self.clear_canvas()

        except Exception as e:
            ui.notify(f'Processing Error: {str(e)}', type='negative')