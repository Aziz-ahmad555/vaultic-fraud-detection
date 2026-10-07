"""
migrate_alerts_nullable.py — one-time migration

Fixes: alerts.transaction_id was NOT NULL, but ml_model.analyze_daily_behavior()
legitimately creates alerts with transaction_id=None (daily-frequency anomalies
aren't tied to a single transaction). This caused an IntegrityError whenever
that code path fired.

SQLite can't ALTER COLUMN directly, so this script:
  1. Renames the existing 'alerts' table to 'alerts_old'
  2. Creates a new 'alerts' table with transaction_id NULLABLE
  3. Copies all existing rows across unchanged
  4. Drops 'alerts_old'

Run this ONCE, with the app NOT running, from your project root:
    python migrate_alerts_nullable.py

It operates on instance/fraud_detection.db by default -- the same file
your Flask app actually uses (see SQLALCHEMY_DATABASE_URI in app.py).
"""

import sqlite3
import os

DB_PATH = os.path.join("instance", "fraud_detection.db")


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Could not find {DB_PATH} -- check you're running this from "
              f"your project root (same folder as app.py).")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Sanity check: make sure 'alerts' exists before we touch anything.
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
    if not cur.fetchone():
        print("No 'alerts' table found -- nothing to migrate.")
        conn.close()
        return

    print("Starting migration...")

    cur.execute("ALTER TABLE alerts RENAME TO alerts_old")

    cur.execute("""
        CREATE TABLE alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id VARCHAR(64) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id),
            transaction_id VARCHAR(64),
            score FLOAT,
            severity VARCHAR(32),
            status VARCHAR(32) DEFAULT 'Pending',
            message VARCHAR(512),
            generated_at DATETIME
        )
    """)

    cur.execute("""
        INSERT INTO alerts (id, alert_id, user_id, transaction_id, score,
                             severity, status, message, generated_at)
        SELECT id, alert_id, user_id, transaction_id, score,
               severity, status, message, generated_at
        FROM alerts_old
    """)

    copied = cur.rowcount
    cur.execute("DROP TABLE alerts_old")

    conn.commit()
    conn.close()

    print(f"Migration complete. {copied} existing alert(s) preserved.")
    print("alerts.transaction_id is now nullable.")


if __name__ == "__main__":
    migrate()
