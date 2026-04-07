from nicegui import ui
from contextlib import contextmanager

@contextmanager
def professional_layout(page_title: str):
    # 1. Apply a cohesive, modern color palette
    ui.colors(primary='#2563eb', secondary='#475569', accent='#3b82f6', positive='#16a34a', negative='#ef4444')
    
    # 2. Set the custom background image globally
    # Note: We assume the image is served from a '/static' route
    ui.query('body').style('''
        background-image: url("/static/image_4fab87.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-color: #f8fafc;
        margin: 0;
    ''')

    # 3. Create a sleek, frosted-glass header
    with ui.header().classes('bg-slate-900/90 backdrop-blur-md text-white p-4 flex justify-between items-center shadow-lg'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('memory', size='md').classes('text-blue-400')
            ui.label('AI Digit Recognizer').classes('text-2xl font-extrabold tracking-tight')
        
        # Consistent Navigation
        with ui.row().classes('items-center gap-2'):
            ui.button('Draw', on_click=lambda: ui.navigate.to('/')).props('flat color=white icon=edit')
            ui.button('History', on_click=lambda: ui.navigate.to('/history')).props('flat color=white icon=history')
            ui.button('Admin', on_click=lambda: ui.navigate.to('/admin')).props('flat color=white icon=admin_panel_settings')
            ui.button('Logout', on_click=lambda: ui.navigate.to('/logout')).props('flat outline color=white icon=logout')

    # 4. Main content container (centered frosted-glass card)
    with ui.column().classes('w-full min-h-[85vh] items-center justify-center p-6 mt-12'):
        with ui.card().classes('w-full max-w-4xl p-8 bg-white/95 backdrop-blur-sm shadow-2xl rounded-2xl flex flex-col items-center gap-6'):
            # Page Title
            ui.label(page_title).classes('text-3xl font-bold text-slate-800 border-b-2 border-blue-500 pb-2 mb-2')
            
            # Yield allows the specific page content to be injected here
            yield