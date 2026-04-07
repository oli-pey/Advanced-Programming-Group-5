import base64
from nicegui import ui, app
from DB.database import SessionLocal, PredictionEntry, User
from web.layout import professional_layout

class AdminDashboard:
    def render(self):
        if not app.storage.user.get("is_admin", False):
            ui.notify("Admin access only.", type="negative")
            ui.navigate.to("/")
            return

        with professional_layout("Admin Dashboard"):
            ui.label("Administrative Control Panel").classes("text-lg text-slate-600 mb-4")
            with ui.row().classes("gap-6"):
                ui.button("New drawing", on_click=lambda: ui.navigate.to("/")).props("color=primary icon=brush")
                ui.button("View all user submissions", on_click=lambda: ui.navigate.to("/admin/history")).props("outline icon=manage_search")


class AdminHistoryPage:
    def render(self):
        if not app.storage.user.get("is_admin", False):
            ui.notify("Admin access only.", type="negative")
            ui.navigate.to("/")
            return

        def load_all_entries():
            db = SessionLocal()
            try:
                entries = (
                    db.query(PredictionEntry, User.username)
                    .join(User, PredictionEntry.user_id == User.id)
                    .order_by(PredictionEntry.created_at.desc())
                    .all()
                )
                rows = []
                for entry, username in entries:
                    original_base64 = base64.b64encode(entry.original_image).decode("utf-8")
                    rows.append({
                        "id": entry.id,
                        "username": username,
                        "prediction": entry.prediction,
                        "original": f"data:image/png;base64,{original_base64}",
                        "date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                return rows
            finally:
                db.close()

        async def admin_delete_entry(entry_id):
            db = SessionLocal()
            try:
                entry = db.query(PredictionEntry).filter(PredictionEntry.id == entry_id).first()
                if entry:
                    db.delete(entry)
                    db.commit()
                    ui.notify(f"Admin: Entry {entry_id} deleted.")
                    table.rows[:] = load_all_entries()
            except Exception as e:
                ui.notify(f"Error: {e}", type='negative')
            finally:
                db.close()

        with professional_layout("Global Submission History"):
            rows = load_all_entries()
            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True},
                {"name": "username", "label": "User", "field": "username", "sortable": True},
                {"name": "original", "label": "Drawing", "field": "original", "align": "center"},
                {"name": "prediction", "label": "Prediction", "field": "prediction", "sortable": True},
                {"name": "date", "label": "Timestamp", "field": "date", "sortable": True},
                {"name": "delete", "label": "Actions", "field": "id"}
            ]

            table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full shadow-lg border")

            table.add_slot("body-cell-original", r'''
            <q-td :props="props">
              <img :src="props.value" alt="drawing" style="width:60px;height:60px;object-fit:contain;" />
            </q-td>
            ''')

            table.add_slot("body-cell-delete", r'''
            <q-td :props="props">
              <q-btn flat round icon="delete" color="red" @click="$parent.$emit('admin_delete', props.value)" />
            </q-td>
            ''')
            
            table.on('admin_delete', lambda msg: admin_delete_entry(msg.args))

            if not rows:
                ui.label("No entries found in database.").classes("mt-10 text-xl text-gray-400 italic")