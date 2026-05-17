import os

from nicegui import ui

from web.routes import register_routes


def main():
    register_routes()

    ui.run(
        title='Digit Recognizer',
        storage_secret=os.getenv(
            'STORAGE_SECRET',
            'dev-secret-change-me',
        ),
    )


if __name__ in {'__main__', '__mp_main__'}:
    main()