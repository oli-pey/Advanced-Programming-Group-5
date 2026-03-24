import base64
from nicegui import ui, app

from DB.database import SessionLocal, PredictionEntry, User


class HistoryPage:
    def render(self):
        user_id = app.storage.user.get("user_id")
        username = app.storage.user.get("username")
        is_admin = app.storage.user.get("is_admin", False)

        if not user_id:
            ui.navigate.to("/login")
            return

        db = SessionLocal()
        try:
            entries = (
                db.query(PredictionEntry)
                .filter(PredictionEntry.user_id == user_id)
                .order_by(PredictionEntry.created_at.desc())
                .all()
            )
        finally:
            db.close()

        rows = []
        for entry in entries:
            original_base64 = base64.b64encode(entry.original_image).decode("utf-8")
            rows.append({
                "id": entry.id,
                "prediction": entry.prediction,
                "original": f"data:image/png;base64,{original_base64}",
                "date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            })

        with ui.header().classes("bg-primary text-white p-4 justify-between items-center"):
            ui.label("My History").classes("text-2xl font-bold")
            with ui.row().classes("gap-2"):
                ui.label(f"User: {username}")
                ui.button("New Drawing", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=white icon=add"
                )
                if is_admin:
                    ui.button("Admin", on_click=lambda: ui.navigate.to("/admin")).props(
                        "flat color=white icon=admin_panel_settings"
                    )

        with ui.column().classes("w-full items-center mt-6 pb-10"):
            ui.label("Your Recognition History").classes("text-3xl font-bold text-gray-800 mb-6")

            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True},
                {"name": "original", "label": "Drawing", "field": "original", "align": "center"},
                {"name": "prediction", "label": "Prediction", "field": "prediction", "sortable": True},
                {"name": "date", "label": "Timestamp", "field": "date", "sortable": True},
            ]

            table = ui.table(columns=columns, rows=rows, row_key="id").classes(
                "w-10/12 shadow-xl border-2"
            )

            table.add_slot("body-cell-original", r'''
            <q-td :props="props">
              <img :src="props.value" alt="drawing" style="width:60px;height:60px;object-fit:contain;" />
            </q-td>
            ''')

            if not rows:
                ui.label("No entries found in your history.").classes(
                    "mt-10 text-xl text-gray-400 italic"
                )
