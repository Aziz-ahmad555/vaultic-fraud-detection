# migrate_phase5.py
import sqlite3

conn = sqlite3.connect('instance/fraud_detection.db')
cur = conn.cursor()

# 1. Add is_frozen to users table
try:
    cur.execute("ALTER TABLE users ADD COLUMN is_frozen BOOLEAN DEFAULT 0")
    print("added is_frozen to users")
except sqlite3.OperationalError as e:
    print(f"skipped is_frozen: {e}")

# 2. Create disputes table (matches models.py Dispute class exactly)
cur.execute("""
CREATE TABLE IF NOT EXISTS disputes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dispute_id VARCHAR(64) UNIQUE NOT NULL,
    alert_id VARCHAR(64) NOT NULL REFERENCES alerts(alert_id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    reason TEXT NOT NULL,
    status VARCHAR(32) DEFAULT 'Open',
    created_at DATETIME
)
""")
print("disputes table ready")

conn.commit()
conn.close()