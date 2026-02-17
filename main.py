from nicegui import ui
from web.index import LandingPage
from web.history import HistoryPage

# Instantiate classes as modular units [cite: 30]
landing = LandingPage()
history = HistoryPage()

@ui.page('/')
def main_page():
    landing.render()

@ui.page('/history')
def history_view():
    history.render()

# The app runs in the browser via NiceGUI [cite: 23]
ui.run(title="Advanced Programming Project - Group 5", port=8080)