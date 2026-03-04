from nicegui import ui

class LandingPage:
    def __init__(self):
        self.title = "Handwritten Digit Recognizer"
        self.path = []  # Stores coordinates of the current stroke
        self.draw_content = "" # Stores the SVG paths for the drawing

    def render(self):
        # Header with Navigation
        with ui.header().classes('bg-primary text-white p-4 justify-between items-center'):
            ui.label(self.title).classes('text-2xl font-bold')
            ui.button('View History', on_click=lambda: ui.navigate.to('/history')).props('flat color=white icon=history')

        # Main Layout
        with ui.column().classes('w-full items-center mt-10 space-y-4'):
            ui.label('Draw a digit (0-9) below:').classes('text-xl')

            # --- UPDATED SIZE AND STYLE ---
        self.ii = ui.interactive_image(
        size=(480, 480), 
        on_mouse=self.handle_mouse, 
        events=['mousedown', 'mousemove', 'mouseup'],
        cross=False
        ).classes('border-4 border-gray-400 bg-white cursor-crosshair shadow-lg') \
        .style('width: 480px; height: 480px;')  # <--- ADD THIS LINE

        with ui.row().classes('mt-4 space-x-4'):
                ui.button('Clear', on_click=self.clear_canvas).props('outline color=red')
                ui.button('Predict', on_click=self.process_drawing).props('color=primary')

    def handle_mouse(self, e):
        # Logic to capture mouse movement and convert to SVG paths
        if e.type == 'mousedown':
            self.path = [(e.image_x, e.image_y)]
        elif e.type == 'mousemove' and e.buttons > 0:
            self.path.append((e.image_x, e.image_y))
            # Create SVG line segments
            svg_path = ' '.join([f'{"M" if i==0 else "L"} {p[0]} {p[1]}' for i, p in enumerate(self.path)])
            # Append new stroke to existing content
            new_stroke = f'<path d="{svg_path}" stroke="black" fill="none" stroke-width="12" stroke-linecap="round" />'
            self.ii.content += new_stroke

    def clear_canvas(self):
        """Reset the drawing area"""
        self.ii.content = ""
        ui.notify("Canvas cleared")

    def process_drawing(self):
        """Placeholder for MNIST Model logic"""
        if not self.ii.content:
            ui.notify("Please draw something first!", type='warning')
            return
            
        ui.notify('Processing drawing for recognition...')
        # In a real app, you would convert the SVG in self.ii.content 
        # to a 28x28 grayscale image for your MNIST model.