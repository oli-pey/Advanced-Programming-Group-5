
from nicegui import ui, app
from auth import bootstrap_defaults
import auth




def main():
    auth.bootstrap_defaults()  # This ensures 'admin' exists with admin rights
    ui.run(storage_secret='PICK_A_SECURE_PASSWORD')



if __name__ in {"__main__", "__mp_main__"}:
    main()