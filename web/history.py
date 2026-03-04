import base64
from nicegui import ui
from DB.database import SessionLocal, PredictionEntry

class HistoryPage:
    def render(self):
        # 1. Persistence Layer: Fetch data using the ORM
        db = SessionLocal()
        try:
            # Fetch all entries ordered by most recent
            entries = db.query(PredictionEntry).order_by(PredictionEntry.created_at.desc()).all()
        finally:
            db.close()

        # 2. Application Logic: Transform ORM objects into UI-ready dictionaries
        rows = []
        for entry in entries:
            # Convert only the original binary PNG data to Base64
            original_base64 = base64.b64encode(entry.original_image).decode('utf-8')
            
            rows.append({
                'id': entry.id,
                'prediction': entry.prediction,
                'original': f'data:image/png;base64,{original_base64}',
                'date': entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })

        # 3. Presentation Layer: Define the UI components without the 28x28 column
        with ui.header().classes('bg-primary text-white p-4 justify-between items-center'):
            ui.label('History Log').classes('text-2xl font-bold')
            ui.button('New Drawing', on_click=lambda: ui.navigate.to('/')).props('flat color=white icon=add')

        with ui.column().classes('w-full items-center mt-6 pb-10'):
            ui.label('Recognition History').classes('text-3xl font-bold text-gray-800 mb-6')
            
            # Updated columns: Removed 'downsized'
            columns = [
                {'name': 'id', 'label': 'ID', 'field': 'id', 'sortable': True},
                {'name': 'original', 'label': 'Drawing', 'field': 'original', 'align': 'center'},
                {'name': 'prediction', 'label': 'Prediction', 'field': 'prediction', 'sortable': True},
                {'name': 'date', 'label': 'Timestamp', 'field': 'date', 'sortable': True},
            ]

            # Render the table
            table = ui.table(columns=columns, rows=rows, row_key='id').classes('w-10/12 shadow-xl border-2')
            
            # Custom slot for the original drawing
            table.add_slot('body-cell-original', '''
                <q-td :props="props">
                    <img :src="props.value" style="height: 80px; border: 1px solid #ccc; border-radius: 4px;" />
                </q-td>
            ''')

            if not rows:
                ui.label('No entries found in database.').classes('mt-10 text-xl text-gray-400 italic')