# migrate_add_columns.py
import sqlite3

conn = sqlite3.connect('instance/fraud_detection.db')
cur = conn.cursor()

for col, coltype in [('scored', 'BOOLEAN DEFAULT 0'),
                      ('scored_at', 'DATETIME'),
                      ('fraud_score', 'FLOAT')]:
    try:
        cur.execute(f"ALTER TABLE transactions ADD COLUMN {col} {coltype}")
        print(f"added {col}")
    except sqlite3.OperationalError as e:
        print(f"skipped {col}: {e}")

conn.commit()
conn.close()