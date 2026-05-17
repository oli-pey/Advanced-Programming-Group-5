from pathlib import Path
import base64

from sandbox_ml.config import TrainingConfig
from sandbox_ml.training import train_sandbox_model
from sandbox_ml.recognizer import SandboxModelRecognizer

from nicegui import ui, app

from DB.database import SessionLocal, SandboxClass, SandboxSample, SandboxTrainedModel
from web.auth import require_session
from sandbox.services import (
    SandboxError,
    create_class,
    create_dataset,
    create_sample,
    delete_class,
    delete_dataset,
    delete_sample,
    get_dataset_for_user,
    list_datasets_for_user,
)
from web.layout import professional_layout


class SandboxOverviewPage:
    def render(self):
        if not require_session():
            return

        current_user_id = app.storage.user.get('user_id')

        with professional_layout('Sandbox Datasets'):
            ui.label(
                'Create personal datasets for custom classifiers. '
                'Phase 1 includes datasets, classes, sample storage, and Phase 2 drawing input.'
            ).classes('text-lg text-slate-600 mb-4')

            with ui.card().classes('w-full p-4 mb-6 bg-slate-50 border border-slate-200'):
                ui.label('Create New Dataset').classes('text-xl font-semibold mb-3')
                name_input = ui.input('Dataset name').classes('w-full').props('outlined')
                desc_input = ui.textarea('Description (optional)').classes('w-full').props('outlined')
                shared_toggle = ui.switch('Share dataset with everyone').classes('mt-2')

                def handle_create_dataset():
                    name = ' '.join((name_input.value or '').strip().split())
                    if not name:
                        ui.notify('Please enter a dataset name.', type='warning')
                        return

                    db = SessionLocal()
                    try:
                        dataset = create_dataset(
                            db=db,
                            owner_user_id=current_user_id,
                            name=name,
                            description=(desc_input.value or '').strip() or None,
                            is_shared=bool(shared_toggle.value),
                        )
                        ui.notify('Dataset created.', type='positive')
                        ui.navigate.to(f'/sandbox/dataset/{dataset.id}')
                    except SandboxError as exc:
                        ui.notify(str(exc), type='negative')
                    finally:
                        db.close()

                ui.button('Create Dataset', on_click=handle_create_dataset).props(
                    'color=primary icon=create_new_folder'
                )

            db = SessionLocal()
            try:
                datasets = list_datasets_for_user(db, current_user_id)
            finally:
                db.close()

            ui.label('My Datasets').classes('text-xl font-semibold mb-2')
            if not datasets:
                ui.label('No datasets yet. Create your first one above.').classes('text-slate-500')
                return

            for dataset in datasets:
                with ui.card().classes('w-full p-4 mb-3 border border-slate-200'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.column().classes('gap-1'):
                            ui.label(dataset.name).classes('text-lg font-semibold')
                            if dataset.description:
                                ui.label(dataset.description).classes('text-slate-600')
                            ui.label(f'Shared: {"Yes" if dataset.is_shared else "No"}').classes(
                                'text-xs text-slate-500'
                            )
                        with ui.row().classes('gap-2'):
                            ui.button(
                                'Open',
                                on_click=lambda d_id=dataset.id: ui.navigate.to(f'/sandbox/dataset/{d_id}'),
                            ).props('color=primary icon=open_in_new')

                            def _delete_dataset(d_id=dataset.id):
                                db = SessionLocal()
                                try:
                                    ds = get_dataset_for_user(db, d_id, current_user_id)
                                    if not ds:
                                        ui.notify('Dataset not found.', type='negative')
                                        return
                                    delete_dataset(db, ds)
                                    ui.notify('Dataset deleted.', type='positive')
                                    ui.navigate.to('/sandbox')
                                finally:
                                    db.close()

                            ui.button('Delete', on_click=_delete_dataset).props(
                                'outline color=negative icon=delete'
                            )


class SandboxDatasetPage:
    def __init__(self, dataset_id: int):
        self.dataset_id = dataset_id

    def render(self):
        if not require_session():
            return

        current_user_id = app.storage.user.get('user_id')
        db = SessionLocal()
        try:
            dataset = get_dataset_for_user(db, self.dataset_id, current_user_id)
            if not dataset:
                with professional_layout('Sandbox Dataset'):
                    ui.label('Dataset not found.').classes('text-red-600')
                return

            dataset_name = dataset.name
            dataset_desc = dataset.description
            dataset_shared = dataset.is_shared
            classes = list(dataset.classes)
            samples = list(dataset.samples)
            class_name_map = {c.id: c.name for c in classes}
        finally:
            db.close()

        with professional_layout(f'Sandbox Dataset: {dataset_name}'):
            ui.button('Back to Sandbox', on_click=lambda: ui.navigate.to('/sandbox')).props(
                'flat icon=arrow_back'
            )

            if dataset_desc:
                ui.label(dataset_desc).classes('text-slate-600')
            ui.label(f'Shared: {"Yes" if dataset_shared else "No"}').classes(
                'text-sm text-slate-500 mb-4'
            )

            with ui.row().classes('w-full gap-6 items-start'):
                with ui.column().classes('w-1/3 gap-4'):
                    with ui.card().classes('w-full p-4 bg-slate-50 border border-slate-200'):
                        ui.label('Add Class').classes('text-lg font-semibold mb-2')
                        class_name = ui.input('Class label').classes('w-full').props('outlined')
                        class_desc = ui.textarea('Description (optional)').classes('w-full').props('outlined')

                        def handle_add_class():
                            name = ' '.join((class_name.value or '').strip().split())
                            if not name:
                                ui.notify('Please enter a class label.', type='warning')
                                return

                            db = SessionLocal()
                            try:
                                dataset = get_dataset_for_user(db, self.dataset_id, current_user_id)
                                if not dataset:
                                    ui.notify('Dataset not found.', type='negative')
                                    return
                                create_class(
                                    db=db,
                                    dataset=dataset,
                                    name=name,
                                    description=(class_desc.value or '').strip() or None,
                                )
                                ui.notify('Class created.', type='positive')
                                ui.navigate.to(f'/sandbox/dataset/{self.dataset_id}')
                            except SandboxError as exc:
                                ui.notify(str(exc), type='negative')
                            finally:
                                db.close()

                        ui.button('Add Class', on_click=handle_add_class).props(
                            'color=primary icon=label'
                        )

                    self._render_upload_sample_card(classes, current_user_id)
                    self._render_draw_sample_card(classes, current_user_id)
                    self._render_training_card(classes,samples, current_user_id)
                    self._render_prediction_card(current_user_id)


                with ui.column().classes('w-2/3 gap-4'):
                    self._render_classes_card(classes, current_user_id)
                    self._render_samples_card(samples, class_name_map, current_user_id)

    def _render_upload_sample_card(self, classes, current_user_id: int):
        with ui.card().classes('w-full p-4 bg-slate-50 border border-slate-200'):
            ui.label('Upload Sample').classes('text-lg font-semibold mb-2')

            if classes:
                class_options = {str(c.id): c.name for c in classes}
                first_class_id = next(iter(class_options.keys()))
                selected_class = ui.select(
                    options=class_options,
                    value=first_class_id,
                    label='Class',
                ).classes('w-full').props('outlined')
            else:
                selected_class = None
                ui.label('Create a class before uploading samples.').classes('text-slate-500')

            note_input = ui.textarea('Note (optional)').classes('w-full').props('outlined')

            uploaded_file_data = {'content': None, 'filename': None}
            file_label = ui.label('No file selected').classes('text-sm text-slate-500')

            async def handle_file_upload(e):
                try:
                    safe_name = Path(e.file.name).name
                    chunks = []
                    async for chunk in e.file.iterate():
                        chunks.append(chunk)
                    content = b''.join(chunks)

                    if not content:
                        ui.notify('Uploaded file is empty.', type='negative')
                        return

                    uploaded_file_data['content'] = content
                    uploaded_file_data['filename'] = safe_name
                    file_label.set_text(f'Selected file: {safe_name}')
                    ui.notify('File loaded.', type='positive')
                except Exception as exc:
                    ui.notify(f'Upload failed: {exc}', type='negative')

            ui.upload(
                label='Choose image file',
                on_upload=handle_file_upload,
                auto_upload=True,
                max_files=1,
            ).classes('w-full').props('accept=.png,.jpg,.jpeg,.bmp,.webp')

            def handle_upload_sample():
                if not selected_class or selected_class.value is None:
                    ui.notify('Please select a class first.', type='warning')
                    return

                if not uploaded_file_data['content'] or not uploaded_file_data['filename']:
                    ui.notify('Please upload an image file first.', type='warning')
                    return

                db = SessionLocal()
                try:
                    dataset = get_dataset_for_user(db, self.dataset_id, current_user_id)
                    if not dataset:
                        ui.notify('Dataset not found.', type='negative')
                        return

                    create_sample(
                        db=db,
                        dataset=dataset,
                        class_id=int(selected_class.value),
                        source_type='uploaded',
                        filename=uploaded_file_data['filename'],
                        content=uploaded_file_data['content'],
                        user_note=(note_input.value or '').strip() or None,
                    )

                    ui.notify('Sample uploaded.', type='positive')
                    ui.navigate.to(f'/sandbox/dataset/{self.dataset_id}')
                except (SandboxError, ValueError) as exc:
                    ui.notify(str(exc), type='negative')
                finally:
                    db.close()

            ui.button('Save Uploaded Sample', on_click=handle_upload_sample).props(
                'color=primary icon=upload'
            )

    def _render_draw_sample_card(self, classes, current_user_id: int):
        with ui.card().classes('w-full p-4 bg-slate-50 border border-slate-200'):
            ui.label('Draw Sample').classes('text-lg font-semibold mb-2')
            ui.label(
                'Draw in white on the black canvas. The 500x500 PNG is stored now; later training will resize it to 64x64.'
            ).classes('text-sm text-slate-500 mb-2')

            if classes:
                class_options = {str(c.id): c.name for c in classes}
                first_class_id = next(iter(class_options.keys()))
                selected_class = ui.select(
                    options=class_options,
                    value=first_class_id,
                    label='Class',
                ).classes('w-full').props('outlined')
            else:
                selected_class = None
                ui.label('Create a class before drawing samples.').classes('text-slate-500')

            note_input = ui.textarea('Note (optional)').classes('w-full').props('outlined')

            canvas_id = f'sandbox_canvas_{self.dataset_id}'
            brush_id = f'sandbox_brush_{self.dataset_id}'

            ui.html(f"""
            <div style="width: 100%; display: flex; flex-direction: column; gap: 8px;">
                <canvas id="{canvas_id}" width="500" height="500"
                    style="width: 100%; max-width: 500px; height: auto; border: 1px solid #94a3b8; border-radius: 8px; background: #000; touch-action: none;">
                </canvas>
                <div style="display:flex; align-items:center; gap:8px;">
                    <label for="{brush_id}" style="font-size: 14px; color: #475569;">Brush size</label>
                    <input id="{brush_id}" type="range" min="4" max="40" value="18" />
                    <span style="font-size: 12px; color: #64748b;">Use mouse or touch</span>
                </div>
            </div>
            """)

            ui.add_body_html(f"""
            <script>
                setTimeout(function() {{
                    const canvas = document.getElementById('{canvas_id}');
                    const brush = document.getElementById('{brush_id}');
                    if (!canvas || !brush) return;

                    const ctx = canvas.getContext('2d');

                    function resetCanvas() {{
                        ctx.fillStyle = 'black';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                    }}

                    resetCanvas();
                    window['clear_{canvas_id}'] = resetCanvas;

                    let drawing = false;
                    let lastX = 0;
                    let lastY = 0;

                    function getPos(event) {{
                        const rect = canvas.getBoundingClientRect();
                        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
                        const clientY = event.touches ? event.touches[0].clientY : event.clientY;
                        return {{
                            x: (clientX - rect.left) * (canvas.width / rect.width),
                            y: (clientY - rect.top) * (canvas.height / rect.height)
                        }};
                    }}

                    function start(event) {{
                        event.preventDefault();
                        drawing = true;
                        const pos = getPos(event);
                        lastX = pos.x;
                        lastY = pos.y;
                    }}

                    function draw(event) {{
                        if (!drawing) return;
                        event.preventDefault();
                        const pos = getPos(event);

                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = Number(brush.value);
                        ctx.lineCap = 'round';
                        ctx.lineJoin = 'round';

                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(pos.x, pos.y);
                        ctx.stroke();

                        lastX = pos.x;
                        lastY = pos.y;
                    }}

                    function stop(event) {{
                        event.preventDefault();
                        drawing = false;
                    }}

                    canvas.addEventListener('mousedown', start);
                    canvas.addEventListener('mousemove', draw);
                    canvas.addEventListener('mouseup', stop);
                    canvas.addEventListener('mouseleave', stop);

                    canvas.addEventListener('touchstart', start, {{passive: false}});
                    canvas.addEventListener('touchmove', draw, {{passive: false}});
                    canvas.addEventListener('touchend', stop, {{passive: false}});
                    canvas.addEventListener('touchcancel', stop, {{passive: false}});
                }}, 100);
            </script>
            """)

            async def clear_canvas():
                await ui.run_javascript(f"window['clear_{canvas_id}']();")

            async def save_drawing():
                if not selected_class or selected_class.value is None:
                    ui.notify('Please select a class first.', type='warning')
                    return

                try:
                    data_url = await ui.run_javascript(
                        f"document.getElementById('{canvas_id}').toDataURL('image/png');",
                        timeout=5.0,
                    )
                    if not data_url or ',' not in data_url:
                        ui.notify('Could not read canvas data.', type='negative')
                        return

                    encoded = data_url.split(',', 1)[1]
                    content = base64.b64decode(encoded)
                except Exception as exc:
                    ui.notify(f'Could not save drawing: {exc}', type='negative')
                    return

                db = SessionLocal()
                try:
                    dataset = get_dataset_for_user(db, self.dataset_id, current_user_id)
                    if not dataset:
                        ui.notify('Dataset not found.', type='negative')
                        return

                    create_sample(
                        db=db,
                        dataset=dataset,
                        class_id=int(selected_class.value),
                        source_type='drawn',
                        filename='drawing.png',
                        content=content,
                        user_note=(note_input.value or '').strip() or None,
                    )
                    ui.notify('Drawing saved as sample.', type='positive')
                    ui.navigate.to(f'/sandbox/dataset/{self.dataset_id}')
                except (SandboxError, ValueError) as exc:
                    ui.notify(str(exc), type='negative')
                finally:
                    db.close()

            with ui.row().classes('gap-2'):
                ui.button('Clear Canvas', on_click=clear_canvas).props('outline icon=delete_sweep')
                ui.button('Save Drawing Sample', on_click=save_drawing).props('color=primary icon=draw')

    def _render_classes_card(self, classes, current_user_id: int):
        with ui.card().classes('w-full p-4 border border-slate-200'):
            ui.label('Classes').classes('text-lg font-semibold mb-2')
            if not classes:
                ui.label('No classes yet.').classes('text-slate-500')
            else:
                for c in classes:
                    with ui.row().classes('w-full items-center justify-between py-1'):
                        with ui.column().classes('gap-0'):
                            ui.label(c.name).classes('font-medium')
                            if c.description:
                                ui.label(c.description).classes('text-sm text-slate-500')

                        def _delete_class(class_id=c.id):
                            db = SessionLocal()
                            try:
                                sandbox_class = db.query(SandboxClass).filter(
                                    SandboxClass.id == class_id,
                                    SandboxClass.dataset.has(owner_user_id=current_user_id),
                                ).first()
                                if not sandbox_class:
                                    ui.notify('Class not found.', type='negative')
                                    return
                                delete_class(db, sandbox_class)
                                ui.notify('Class deleted.', type='positive')
                                ui.navigate.to(f'/sandbox/dataset/{self.dataset_id}')
                            finally:
                                db.close()

                        ui.button('Delete', on_click=_delete_class).props(
                            'flat color=negative icon=delete'
                        )

    def _render_samples_card(self, samples, class_name_map, current_user_id: int):
        with ui.card().classes('w-full p-4 border border-slate-200'):
            ui.label('Samples').classes('text-lg font-semibold mb-2')
            if not samples:
                ui.label('No samples yet.').classes('text-slate-500')
            else:
                with ui.grid(columns=2).classes('w-full gap-4'):
                    for s in samples:
                        with ui.card().classes('p-3 border border-slate-200'):
                            if s.image_data:
                                encoded = base64.b64encode(s.image_data).decode('utf-8')
                                mime_type = s.image_mime_type or 'image/png'

                                ui.image(f'data:{mime_type};base64,{encoded}').classes(
                                    'w-full h-48 object-contain bg-slate-50 rounded'
                                )
                            else:
                                ui.label('Image data missing.').classes('text-red-500')

                            ui.label(
                                f'Class: {class_name_map.get(s.class_id, "Unknown")}'
                            ).classes('font-medium')
                            ui.label(f'Source: {s.source_type}').classes('text-sm text-slate-500')
                            if s.user_note:
                                ui.label(s.user_note).classes('text-sm text-slate-600')

                            def _delete_sample(sample_id=s.id):
                                db = SessionLocal()
                                try:
                                    sample = db.query(SandboxSample).filter(
                                        SandboxSample.id == sample_id,
                                        SandboxSample.dataset.has(owner_user_id=current_user_id),
                                    ).first()
                                    if not sample:
                                        ui.notify('Sample not found.', type='negative')
                                        return
                                    delete_sample(db, sample)
                                    ui.notify('Sample deleted.', type='positive')
                                    ui.navigate.to(f'/sandbox/dataset/{self.dataset_id}')
                                finally:
                                    db.close()

                            ui.button('Delete', on_click=_delete_sample).props(
                                'outline color=negative icon=delete'
                            )

    def _render_training_card(self, classes, samples, current_user_id: int):
        with ui.card().classes('w-full p-4 bg-slate-50 border border-slate-200'):
            ui.label('Train Custom Model').classes('text-lg font-semibold mb-2')
            ui.label('Requires at least 2 classes and at least 5 samples per class.').classes(
                'text-sm text-slate-500'
            )

            model_type = ui.select(
                options=['cnn', 'mlp', 'logreg'],
                value='cnn',
                label='Model type',
            ).classes('w-full').props('outlined')

            model_name = ui.input('Model name (optional)').classes('w-full').props('outlined')

            epochs = ui.number('Epochs', value=10, min=1, max=100).classes('w-full').props('outlined')
            batch_size = ui.number('Batch size', value=16, min=1, max=256).classes('w-full').props('outlined')
            learning_rate = ui.number(
                'Learning rate',
                value=0.001,
                min=0.00001,
                max=1.0,
                step=0.0001,
            ).classes('w-full').props('outlined')

            def handle_train():
                config = TrainingConfig(
                    epochs=int(epochs.value),
                    batch_size=int(batch_size.value),
                    learning_rate=float(learning_rate.value),
                )

                try:
                    ui.notify('Training started. The page may pause briefly.', type='info')

                    trained_model = train_sandbox_model(
                        dataset_id=self.dataset_id,
                        owner_user_id=current_user_id,
                        model_type=model_type.value,
                        model_name=(model_name.value or '').strip() or None,
                        config=config,
                    )

                    ui.notify(f'Model trained successfully: {trained_model.name}', type='positive')
                    ui.navigate.to(f'/sandbox/dataset/{self.dataset_id}')

                except Exception as exc:
                    ui.notify(f'Training failed: {exc}', type='negative')

            ui.button('Train Model', on_click=handle_train).props('color=primary icon=model_training')

    def _render_prediction_card(self, current_user_id: int):
        db = SessionLocal()
        try:
            trained_models = (
                db.query(SandboxTrainedModel)
                .filter(
                    SandboxTrainedModel.dataset_id == self.dataset_id,
                    SandboxTrainedModel.owner_user_id == current_user_id,
                )
                .order_by(SandboxTrainedModel.created_at.desc())
                .all()
            )
        finally:
            db.close()

        with ui.card().classes('w-full p-4 bg-slate-50 border border-slate-200'):
            ui.label('Predict with Trained Model').classes('text-lg font-semibold mb-2')

            if not trained_models:
                ui.label('No trained models yet. Train a model first.').classes('text-slate-500')
                return

            model_options = {
                str(m.id): f'{m.name} ({m.model_type})'
                for m in trained_models
            }

            selected_model = ui.select(
                options=model_options,
                value=next(iter(model_options.keys())),
                label='Trained model',
            ).classes('w-full').props('outlined')

            result_area = ui.column().classes('w-full gap-1')

            uploaded_file_data = {'content': None, 'filename': None}
            file_label = ui.label('No prediction image selected').classes('text-sm text-slate-500')

            async def handle_prediction_upload(e):
                try:
                    safe_name = Path(e.file.name).name
                    chunks = []
                    async for chunk in e.file.iterate():
                        chunks.append(chunk)
                    content = b''.join(chunks)

                    if not content:
                        ui.notify('Uploaded file is empty.', type='negative')
                        return

                    uploaded_file_data['content'] = content
                    uploaded_file_data['filename'] = safe_name
                    file_label.set_text(f'Selected file: {safe_name}')
                    ui.notify('Prediction image loaded.', type='positive')
                except Exception as exc:
                    ui.notify(f'Upload failed: {exc}', type='negative')

            ui.upload(
                label='Upload image for prediction',
                on_upload=handle_prediction_upload,
                auto_upload=True,
                max_files=1,
            ).classes('w-full').props('accept=.png,.jpg,.jpeg,.bmp,.webp')

            def render_prediction_result(result: dict):
                result_area.clear()
                with result_area:
                    ui.label(f'Predicted label: {result["predicted_label"]}').classes(
                        'text-lg font-semibold text-green-700'
                    )
                    ui.label(f'Confidence: {result["confidence"]:.2%}').classes(
                        'text-sm text-slate-600'
                    )
                    ui.label('Class probabilities').classes('font-medium mt-2')

                    for label, prob in sorted(
                            result['probabilities'].items(),
                            key=lambda item: item[1],
                            reverse=True,
                    ):
                        ui.label(f'{label}: {prob:.2%}').classes('text-sm text-slate-600')

            def get_selected_model_record():
                db = SessionLocal()
                try:
                    return (
                        db.query(SandboxTrainedModel)
                        .filter(
                            SandboxTrainedModel.id == int(selected_model.value),
                            SandboxTrainedModel.owner_user_id == current_user_id,
                        )
                        .first()
                    )
                finally:
                    db.close()

            def predict_uploaded_image():
                if not uploaded_file_data['content']:
                    ui.notify('Please upload an image first.', type='warning')
                    return

                model_record = get_selected_model_record()
                if not model_record:
                    ui.notify('Trained model not found.', type='negative')
                    return

                try:
                    recognizer = SandboxModelRecognizer(model_record.checkpoint_path)
                    result = recognizer.predict_from_image_bytes(uploaded_file_data['content'])
                    render_prediction_result(result)
                except Exception as exc:
                    ui.notify(f'Prediction failed: {exc}', type='negative')

            ui.button('Predict Uploaded Image', on_click=predict_uploaded_image).props(
                'color=primary icon=psychology'
            )

            ui.separator().classes('my-4')
            ui.label('Or draw an image for prediction').classes('font-medium')

            canvas_id = f'prediction_canvas_{self.dataset_id}'
            brush_id = f'prediction_brush_{self.dataset_id}'

            ui.html(f'''
            <div style="width: 100%; display: flex; flex-direction: column; gap: 8px;">
                <canvas id="{canvas_id}" width="500" height="500"
                    style="width: 100%; max-width: 500px; height: auto; border: 1px solid #94a3b8; border-radius: 8px; background: #000; touch-action: none;">
                </canvas>
                <div style="display:flex; align-items:center; gap:8px;">
                    <label for="{brush_id}" style="font-size: 14px; color: #475569;">Brush size</label>
                    <input id="{brush_id}" type="range" min="4" max="40" value="18" />
                </div>
            </div>
            ''')

            ui.add_body_html(f'''
            <script>
                setTimeout(function() {{
                    const canvas = document.getElementById('{canvas_id}');
                    const brush = document.getElementById('{brush_id}');
                    if (!canvas || !brush) return;

                    const ctx = canvas.getContext('2d');

                    function resetCanvas() {{
                        ctx.fillStyle = 'black';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                    }}

                    resetCanvas();
                    window['clear_{canvas_id}'] = resetCanvas;

                    let drawing = false;
                    let lastX = 0;
                    let lastY = 0;

                    function getPos(event) {{
                        const rect = canvas.getBoundingClientRect();
                        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
                        const clientY = event.touches ? event.touches[0].clientY : event.clientY;
                        return {{
                            x: (clientX - rect.left) * (canvas.width / rect.width),
                            y: (clientY - rect.top) * (canvas.height / rect.height)
                        }};
                    }}

                    function start(event) {{
                        event.preventDefault();
                        drawing = true;
                        const pos = getPos(event);
                        lastX = pos.x;
                        lastY = pos.y;
                    }}

                    function draw(event) {{
                        if (!drawing) return;
                        event.preventDefault();
                        const pos = getPos(event);

                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = Number(brush.value);
                        ctx.lineCap = 'round';
                        ctx.lineJoin = 'round';

                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(pos.x, pos.y);
                        ctx.stroke();

                        lastX = pos.x;
                        lastY = pos.y;
                    }}

                    function stop(event) {{
                        event.preventDefault();
                        drawing = false;
                    }}

                    canvas.addEventListener('mousedown', start);
                    canvas.addEventListener('mousemove', draw);
                    canvas.addEventListener('mouseup', stop);
                    canvas.addEventListener('mouseleave', stop);

                    canvas.addEventListener('touchstart', start, {{passive: false}});
                    canvas.addEventListener('touchmove', draw, {{passive: false}});
                    canvas.addEventListener('touchend', stop, {{passive: false}});
                    canvas.addEventListener('touchcancel', stop, {{passive: false}});
                }}, 100);
            </script>
            ''')

            async def clear_prediction_canvas():
                await ui.run_javascript(f"window['clear_{canvas_id}']();")

            async def predict_drawing():
                try:
                    data_url = await ui.run_javascript(
                        f"document.getElementById('{canvas_id}').toDataURL('image/png');",
                        timeout=5.0,
                    )
                    if not data_url or ',' not in data_url:
                        ui.notify('Could not read canvas data.', type='negative')
                        return

                    content = base64.b64decode(data_url.split(',', 1)[1])
                except Exception as exc:
                    ui.notify(f'Could not read drawing: {exc}', type='negative')
                    return

                model_record = get_selected_model_record()
                if not model_record:
                    ui.notify('Trained model not found.', type='negative')
                    return

                try:
                    recognizer = SandboxModelRecognizer(model_record.checkpoint_path)
                    result = recognizer.predict_from_image_bytes(content)
                    render_prediction_result(result)
                except Exception as exc:
                    ui.notify(f'Prediction failed: {exc}', type='negative')

            with ui.row().classes('gap-2'):
                ui.button('Clear Prediction Canvas', on_click=clear_prediction_canvas).props(
                    'outline icon=delete_sweep'
                )
                ui.button('Predict Drawing', on_click=predict_drawing).props(
                    'color=primary icon=draw'
                )


