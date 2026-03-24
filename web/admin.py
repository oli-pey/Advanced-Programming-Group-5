import base64
from nicegui import ui, app

from DB.database import SessionLocal, PredictionEntry, User


class AdminDashboard:
    def render(self):
        if not app.storage.user.get("is_admin", False):
            ui.notify("Admin access only.", type="negative")
            ui.navigate.to("/")
            return

        with ui.header().classes("bg-primary text-white p-4 justify-between items-center"):
            ui.label("Admin Dashboard").classes("text-2xl font-bold")
            with ui.row().classes("gap-2"):
                ui.button("Draw", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=white icon=brush"
                )
                ui.button("All History", on_click=lambda: ui.navigate.to("/admin/history")).props(
                    "flat color=white icon=manage_search"
                )

        with ui.column().classes("w-full items-center mt-10 gap-4"):
            ui.label("Admin actions").classes("text-3xl font-bold")
            ui.button("Create drawing / prediction", on_click=lambda: ui.navigate.to("/")).props(
                "color=primary size=lg"
            )
            ui.button("View all user submissions", on_click=lambda: ui.navigate.to("/admin/history")).props(
                "outline size=lg"
            )


class AdminHistoryPage:
    def render(self):
        if not app.storage.user.get("is_admin", False):
            ui.notify("Admin access only.", type="negative")
            ui.navigate.to("/")
            return

        db = SessionLocal()
        try:
            entries = (
                db.query(PredictionEntry, User.username)
                .join(User, PredictionEntry.user_id == User.id)
                .order_by(PredictionEntry.created_at.desc())
                .all()
            )
        finally:
            db.close()

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

        with ui.header().classes("bg-primary text-white p-4 justify-between items-center"):
            ui.label("Admin History").classes("text-2xl font-bold")
            with ui.row().classes("gap-2"):
                ui.button("Dashboard", on_click=lambda: ui.navigate.to("/admin")).props(
                    "flat color=white icon=dashboard"
                )
                ui.button("Draw", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=white icon=brush"
                )

        with ui.column().classes("w-full items-center mt-6 pb-10"):
            ui.label("All Recognition History").classes("text-3xl font-bold text-gray-800 mb-6")

            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True},
                {"name": "username", "label": "User", "field": "username", "sortable": True},
                {"name": "original", "label": "Drawing", "field": "original", "align": "center"},
                {"name": "prediction", "label": "Prediction", "field": "prediction", "sortable": True},
                {"name": "date", "label": "Timestamp", "field": "date", "sortable": True},
            ]

            table = ui.table(columns=columns, rows=rows, row_key="id").classes(
                "w-11/12 shadow-xl border-2"
            )

            table.add_slot("body-cell-original", r'''
            <q-td :props="props">
              <img :src="props.value" alt="drawing" style="width:60px;height:60px;object-fit:contain;" />
            </q-td>
            ''')

            if not rows:
                ui.label("No entries found in database.").classes(
                    "mt-10 text-xl text-gray-400 italic"
                )
