import os
import mysql.connector
from dotenv import load_dotenv

# Load values from .env
load_dotenv()

host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
database = os.getenv("DB_NAME")

print("🔍 Using DB config:", host, user, database)

try:
    # Connect
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor()

    # Insert a dummy row into predictions
    cursor.execute("""
        INSERT INTO predictions (patient_name, age, gender, image_url, mask_url, brain_plane, tumor_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, ("Test Patient", 30, "M", "http://example.com/img.jpg", "http://example.com/mask.jpg", "Axial", "Glioma"))

    conn.commit()
    print("✅ Inserted dummy row!")

    # Fetch last 3 rows
    cursor.execute("SELECT id, patient_name, tumor_type, created_at FROM predictions ORDER BY id DESC LIMIT 3")
    for row in cursor.fetchall():
        print(row)

    cursor.close()
    conn.close()

except Exception as e:
    print("❌ DB connection failed:", e)
