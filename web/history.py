import base64

from nicegui import app, ui

from DB.database import PredictionEntry, SessionLocal
from web.layout import professional_layout


class HistoryPage:
    """
    A page that displays the current user's personal prediction history.
    Users can view their past drawings, model results, and delete entries.
    """

    def render(self):
        """Render the personal history table UI."""
        user_id = app.storage.user.get("user_id")

        # Security check: Ensure user is logged in
        if not user_id:
            ui.navigate.to("/login")
            return

        def load_entries():
            """
            Fetches all prediction entries for the logged-in user from the DB.
            Converts binary image data to base64 strings for browser display.
            """
            db = SessionLocal()
            try:
                # Filter by user_id so users can't see others' data
                entries = (
                    db.query(PredictionEntry)
                    .filter(PredictionEntry.user_id == user_id)
                    .order_by(PredictionEntry.created_at.desc())
                    .all()
                )

                rows = []
                for entry in entries:
                    # Convert BLOB to base64 string
                    img_b64 = base64.b64encode(
                        entry.original_image
                    ).decode("utf-8")
                    rows.append({
                        "id": entry.id,
                        "prediction": entry.prediction,
                        "model_name": entry.model_name,
                        "probability": entry.probability,
                        "original": f"data:image/png;base64,{img_b64}",
                        "date": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                return rows
            finally:
                db.close()

        async def delete_entry(entry_id: int):
            """
            Permanently deletes a specific entry.
            Verifies ownership (user_id) again before deleting to prevent
            ID-manipulation attacks.
            """
            db = SessionLocal()
            try:
                # Critical security filter: filter by both entry ID and
                # owner ID
                entry = db.query(PredictionEntry).filter(
                    PredictionEntry.id == entry_id,
                    PredictionEntry.user_id == user_id
                ).first()

                if entry:
                    db.delete(entry)
                    db.commit()
                    ui.notify(f"Entry {entry_id} deleted.", type='info')

                    # Refresh the table display by updating the rows attribute
                    table.rows[:] = load_entries()
                else:
                    ui.notify(
                        "Delete failed: Entry not found or unauthorized.",
                        type='negative'
                    )
            except Exception as e:
                ui.notify(f"Error: {e}", type='negative')
            finally:
                db.close()

        with professional_layout("My History"):
            rows = load_entries()

            # Table column definitions
            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True},
                {
                    "name": "original",
                    "label": "Drawing",
                    "field": "original",
                    "align": "center"
                },
                {
                    "name": "prediction",
                    "label": "Result",
                    "field": "prediction",
                    "sortable": True
                },
                {
                    "name": "model_name",
                    "label": "Model",
                    "field": "model_name",
                    "sortable": True
                },
                {
                    "name": "probability",
                    "label": "Confidence",
                    "field": "probability",
                    "sortable": True
                },
                {
                    "name": "date",
                    "label": "Date",
                    "field": "date",
                    "sortable": True
                },
                {"name": "delete", "label": "Actions", "field": "id"}
            ]

            # Main Data Table
            table = ui.table(
                columns=columns,
                rows=rows,
                row_key="id"
            ).classes("w-full shadow-sm border border-slate-200 rounded-lg")

            # UI Slot: Render the image preview
            table.add_slot("body-cell-original", r'''
                <q-td :props="props">
                    <img :src="props.value"
                         style="width:50px;height:50px;object-fit:contain;"
                         class="rounded border bg-white" />
                </q-td>
            ''')

            # UI Slot: Render the delete button and emit a Vue event
            table.add_slot("body-cell-delete", r'''
                <q-td :props="props">
                    <q-btn flat round icon="delete" color="red"
                           @click="$parent.$emit('delete', props.value)" />
                </q-td>
            ''')

            # Event Bridge: Listen for delete signal from Vue template
            table.on('delete', lambda msg: delete_entry(msg.args))

            # Empty state helper
            if not rows:
                ui.label(
                    "You haven't made any predictions yet."
                ).classes("mt-10 text-center text-slate-400 italic")
