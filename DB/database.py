import sqlite3
from datetime import datetime

# ---------- CREATE DATABASE ----------
conn = sqlite3.connect("mydata.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image BLOB NOT NULL,
    text TEXT,
    created_at TEXT
)
""")

conn.commit()


# ---------- FUNCTION TO INSERT DATA ----------
def insert_entry(image_path, text_value):
    with open(image_path, "rb") as file:
        image_data = file.read()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO entries (image, text, created_at)
    VALUES (?, ?, ?)
    """, (image_data, text_value, timestamp))

    conn.commit()
    print("Entry saved!")


# ---------- FUNCTION TO READ DATA ----------
def get_entries():
    cursor.execute("SELECT id, text, created_at FROM entries")
    rows = cursor.fetchall()

    for row in rows:
        print(row)


# ---------- FUNCTION TO EXPORT IMAGE ----------
def export_image(entry_id, output_path):
    cursor.execute("SELECT image FROM entries WHERE id = ?", (entry_id,))
    result = cursor.fetchone()

    if result:
        with open(output_path, "wb") as file:
            file.write(result[0])
        print("Image exported!")
    else:
        print("Entry not found.")


# ---------- EXAMPLE USAGE ----------
insert_entry("example.jpg", "My first image")
get_entries()

# export_image(1, "output.jpg")  # uncomment to export image


# ---------- CLOSE CONNECTION ----------
conn.close()
