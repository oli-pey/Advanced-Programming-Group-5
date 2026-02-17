from nicegui import ui

class HistoryPage:
    def render(self):
        with ui.header().classes('bg-secondary text-white p-4 items-center'):
            ui.button(on_click=lambda: ui.navigate.to('/')).props('flat color=white icon=arrow_back')
            ui.label('Recognition History').classes('text-2xl font-bold ml-4')

        with ui.column().classes('w-full items-center mt-10'):
            ui.label('Past Submissions (SQLite Data)').classes('text-lg')
            
            # This area will show data stored in the SQLite database 
            with ui.card().classes('w-3/4 p-4'):
                ui.label('No history entries found.').classes('italic text-gray-500')