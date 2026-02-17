from nicegui import ui, events

class LandingPage:
    def __init__(self):
        self.title = "Image Recognition App"

    def render(self):
        # Header with Navigation
        with ui.header().classes('bg-primary text-white p-4 justify-between items-center'):
            ui.label(self.title).classes('text-2xl font-bold')
            ui.button('View History', on_click=lambda: ui.navigate.to('/history')).props('flat color=white icon=history')

        # Main Layout
        with ui.column().classes('w-full items-center mt-10 space-y-4'):
            ui.label('Upload a handwritten image (PNG/JPG):').classes('text-xl')

            # The Upload Component - State resides on the server
            self.uploader = ui.upload(
                label="Image Recognition App",
                on_upload=self.handle_upload,
                auto_upload=True
            ).classes('w-80')

    def handle_upload(self, e: events.UploadEventArguments):
        """Application Logic: Processing the uploaded file"""
        # Read image bytes
        content = e.content.read()
        ui.notify(f'Uploaded {e.name}, processing recognition...')
        
        # Placeholder for MNIST Model logic and Persistence Layer save
        # logic.predict(content)