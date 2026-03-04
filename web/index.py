import io
import datetime
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from nicegui import ui
from DB.database import SessionLocal, PredictionEntry

class LandingPage:
    def __init__(self):
        self.title = "Handwritten Digit Recognizer"
        self.path = [] 

    def render(self):
        with ui.header().classes('bg-primary text-white p-4 justify-between items-center'):
            ui.label(self.title).classes('text-2xl font-bold')
            ui.button('History', on_click=lambda: ui.navigate.to('/history')).props('flat color=white icon=history')

        with ui.column().classes('w-full items-center mt-10 space-y-4'):
            ui.label('Draw a digit (0-9) below:').classes('text-xl')

            self.ii = ui.interactive_image(
                size=(500, 500), 
                on_mouse=self.handle_mouse, 
                events=['mousedown', 'mousemove', 'mouseup'],
                cross=False
            ).classes('border-4 border-gray-400 bg-white cursor-crosshair shadow-lg') \
             .style('width: 500px; height: 500px;')

            with ui.row().classes('mt-4 space-x-4'):
                ui.button('Clear', on_click=self.clear_canvas).props('outline color=red')
                ui.button('Predict & Save', on_click=self.process_drawing).props('color=primary')

    def handle_mouse(self, e):
        if e.type == 'mousedown':
            self.path = [(e.image_x, e.image_y)]
        elif e.type == 'mousemove' and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))
            svg_path = ' '.join([f'{"M" if i==0 else "L"} {p[0]} {p[1]}' for i, p in enumerate(self.path)])
            new_stroke = f'<path d="{svg_path}" stroke="black" fill="none" stroke-width="20" stroke-linecap="round" />'
            self.ii.content += new_stroke

    def clear_canvas(self):
        self.ii.content = ""

    async def process_drawing(self):
        """Application Logic: Using svglib for pure-Python conversion"""
        if not self.ii.content:
            ui.notify("Please draw something first!", type='warning')
            return

        try:
            # 1. Prepare SVG as a file-like object
            full_svg = f'<svg width="500" height="500" xmlns="http://www.w3.org/2000/svg">{self.ii.content}</svg>'
            svg_file = io.BytesIO(full_svg.encode('utf-8'))

            # 2. System Operations: SVG -> Original PNG (Pure Python)
            # svg2rlg reads the SVG and creates a ReportLab drawing
            drawing = svg2rlg(svg_file)
            
            # renderPM.drawToString creates PNG bytes
            original_png_bytes = renderPM.drawToString(drawing, fmt="PNG")

            # 3. System Operations: High-res -> 28x28 Grayscale PNG
            img = Image.open(io.BytesIO(original_png_bytes)).convert('L') 
            img_small = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            small_buffer = io.BytesIO()
            img_small.save(small_buffer, format="PNG")
            downsized_png_bytes = small_buffer.getvalue()

            # 4. Persistence Layer: Save using ORM
            db = SessionLocal()
            try:
                new_entry = PredictionEntry(
                    original_image=original_png_bytes,
                    downsized_image=downsized_png_bytes,
                    prediction="7", # Placeholder for ML logic
                    created_at=datetime.datetime.utcnow()
                )
                db.add(new_entry)
                db.commit()
                ui.notify('Saved successfully using pure-Python libraries!', type='positive')
                self.clear_canvas()
            finally:
                db.close()

        except Exception as e:
            ui.notify(f'Processing Error: {str(e)}', type='negative')