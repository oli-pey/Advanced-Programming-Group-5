import base64
from nicegui import ui, app
from DB.database import SessionLocal, PredictionEntry
from web.layout import professional_layout

class HistoryPage:
    def render(self):
        user_id = app.storage.user.get("user_id")
        if not user_id:
            ui.navigate.to("/login")
            return

        def load_entries():
            db = SessionLocal()
            try:
                entries = db.query(PredictionEntry).filter(PredictionEntry.user_id == user_id).order_by(PredictionEntry.created_at.desc()).all()
                rows = []
                for entry in entries:
                    img_str = base64.b64encode(entry.original_image).decode("utf-8")
                    rows.append({
                        "id": entry.id,
                        "prediction": entry.prediction,
                        "model_name": entry.model_name,    
                        "probability": entry.probability,
                        "original": f"data:image/png;base64,{img_str}",
                        "date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                return rows
            finally:
                db.close()

        with professional_layout("My History"):
            rows = load_entries()
            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True},
                {"name": "original", "label": "Drawing", "field": "original", "align": "center"},
                {"name": "prediction", "label": "Result", "field": "prediction", "sortable": True},
                {"name": "model_name", "label": "Model", "field": "model_name", "sortable": True},
                {"name": "probability", "label": "Confidence", "field": "probability", "sortable": True},
                {"name": "date", "label": "Date", "field": "date", "sortable": True},
                {"name": "delete", "label": "Actions", "field": "id"}
            ]
            
            table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")
            
            table.add_slot("body-cell-original", r'''
                <q-td :props="props"><img :src="props.value" style="width:50px;height:50px;" /></q-td>
            ''')
            table.add_slot("body-cell-delete", r'''
                <q-td :props="props"><q-btn flat round icon="delete" color="red" @click="$parent.$emit('delete', props.value)" /></q-td>
            ''')
            table.on('delete', lambda msg: ui.notify(f"Delete ID: {msg.args}")) # Add deletion logic here