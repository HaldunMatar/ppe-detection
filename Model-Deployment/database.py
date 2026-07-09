import sqlite3
from datetime import datetime
import os
import re

DATABASE = 'detections.db'
# main table holding one row per recorded PPE violation
VIOLATIONS_TABLE = 'ppe_violations'


def _extract_person_id(instance_id, stored):
    """Person id: use the stored value, else parse it from a record id like MM_DD_YYYY_P3_7."""
    if stored is not None:
        return stored
    match = re.search(r'_P(\d+)_', instance_id or '')
    return int(match.group(1)) if match else None


class Database:
    """Database handler for detections and alerts"""

    def __init__(self):
        self.db_path = DATABASE

    def init_db(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # migrate the legacy 'instances' table name to 'ppe_violations' (keeps all data)
            try:
                c.execute(f'ALTER TABLE instances RENAME TO {VIOLATIONS_TABLE}')
            except sqlite3.OperationalError:
                pass  # already renamed, or no legacy table

            c.execute(f'''
                CREATE TABLE IF NOT EXISTS {VIOLATIONS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT UNIQUE,
                    first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_compliant BOOLEAN DEFAULT 0,
                    missing_ppe TEXT,
                    detected_ppe TEXT,
                    person_id INTEGER,
                    snapshot_path TEXT
                )
            ''')

            # add columns to databases created before they existed
            for column, coltype in (('person_id', 'INTEGER'), ('snapshot_path', 'TEXT')):
                try:
                    c.execute(f'ALTER TABLE {VIOLATIONS_TABLE} ADD COLUMN {column} {coltype}')
                except sqlite3.OperationalError:
                    pass  # column already exists

            c.execute('''
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT,
                    snapshot_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (instance_id) REFERENCES ppe_violations(instance_id)
                )
            ''')

            # backfill snapshot_path for older records from their first snapshot
            c.execute(f'''
                UPDATE {VIOLATIONS_TABLE}
                SET snapshot_path = (
                    SELECT snapshot_path FROM snapshots
                    WHERE snapshots.instance_id = {VIOLATIONS_TABLE}.instance_id
                    ORDER BY id ASC LIMIT 1
                )
                WHERE snapshot_path IS NULL
            ''')

            c.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    description TEXT,
                    snapshot_path TEXT
                )
            ''')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error initializing database: {e}")

    def log_instance_snapshot(self, instance_id, missing_ppe, detected_ppe, snapshot_path, person_id=None):
        """Log a snapshot for a violation record"""
        try:
            if not instance_id or not snapshot_path:
                return False

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute(f'SELECT id FROM {VIOLATIONS_TABLE} WHERE instance_id = ?', (instance_id,))
            if not c.fetchone():
                c.execute(f'''
                    INSERT INTO {VIOLATIONS_TABLE} (instance_id, is_compliant, missing_ppe, detected_ppe, person_id, snapshot_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (instance_id, False, ','.join(missing_ppe), ','.join(detected_ppe), person_id, snapshot_path))
                print(f"Created new violation record: {instance_id}")
            else:
                c.execute(f'''
                    UPDATE {VIOLATIONS_TABLE}
                    SET last_updated = CURRENT_TIMESTAMP, missing_ppe = ?, detected_ppe = ?
                    WHERE instance_id = ?
                ''', (','.join(missing_ppe), ','.join(detected_ppe), instance_id))

            c.execute('''
                INSERT INTO snapshots (instance_id, snapshot_path)
                VALUES (?, ?)
            ''', (instance_id, snapshot_path))

            conn.commit()
            conn.close()
            print(f"Logged snapshot for {instance_id}: {snapshot_path}")
            return True
        except Exception as e:
            print(f"Error logging violation snapshot: {e}")
            return False

    def log_alert(self, alert_type, description, snapshot_path):
        """Log an alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO alerts (alert_type, description, snapshot_path)
                VALUES (?, ?, ?)
            ''', (alert_type, description, snapshot_path))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging alert: {e}")

    def get_statistics(self):
        """Get detection statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute(f'SELECT COUNT(*) FROM {VIOLATIONS_TABLE} WHERE is_compliant = 0')
            non_compliant = c.fetchone()[0]

            c.execute('SELECT COUNT(*) FROM alerts')
            total_alerts = c.fetchone()[0]

            conn.close()

            return {
                'total_detections': non_compliant,
                'non_compliant_count': non_compliant,
                'total_alerts': total_alerts
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_detections': 0,
                'non_compliant_count': 0,
                'total_alerts': 0
            }

    def get_all_instances(self, sort_by='first_detected', sort_order='desc'):
        """Get all violation records"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            query = f'''
                SELECT v.instance_id, v.first_detected, v.last_updated, v.is_compliant,
                       v.missing_ppe, v.detected_ppe, v.person_id, v.snapshot_path,
                       COUNT(s.id) as snapshot_count
                FROM {VIOLATIONS_TABLE} v
                LEFT JOIN snapshots s ON v.instance_id = s.instance_id
                WHERE v.is_compliant = 0
                GROUP BY v.instance_id
            '''

            query += f' ORDER BY v.{sort_by} {sort_order}'

            c.execute(query)
            rows = c.fetchall()
            conn.close()

            instances = []
            for row in rows:
                instances.append({
                    'instance_id': row[0],
                    'first_detected': row[1],
                    'last_updated': row[2],
                    'is_compliant': bool(row[3]),
                    'missing_ppe': row[4].split(',') if row[4] else [],
                    'detected_ppe': row[5].split(',') if row[5] else [],
                    'person_id': _extract_person_id(row[0], row[6]),
                    'snapshot_path': row[7],
                    'snapshot_count': row[8]
                })

            return instances
        except Exception as e:
            print(f"Error getting violation records: {e}")
            return []

    def get_instance_snapshots(self, instance_id):
        """Get all snapshots for a specific violation record"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute(f'SELECT * FROM {VIOLATIONS_TABLE} WHERE instance_id = ?', (instance_id,))
            instance_row = c.fetchone()

            if not instance_row:
                conn.close()
                return None

            c.execute('''
                SELECT snapshot_path, timestamp
                FROM snapshots
                WHERE instance_id = ?
                ORDER BY timestamp ASC
            ''', (instance_id,))
            snapshot_rows = c.fetchall()

            conn.close()

            return {
                'instance_id': instance_row[1],
                'first_detected': instance_row[2],
                'last_updated': instance_row[3],
                'missing_ppe': instance_row[5].split(',') if instance_row[5] else [],
                'detected_ppe': instance_row[6].split(',') if instance_row[6] else [],
                'person_id': _extract_person_id(instance_row[1],
                                                instance_row[7] if len(instance_row) > 7 else None),
                'snapshot_path': instance_row[8] if len(instance_row) > 8 else None,
                'snapshots': [{'path': row[0], 'timestamp': row[1]} for row in snapshot_rows]
            }
        except Exception as e:
            print(f"Error getting violation snapshots: {e}")
            return None

    def delete_instance(self, instance_id):
        """Delete a violation record and all its snapshots"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute('SELECT snapshot_path FROM snapshots WHERE instance_id = ?', (instance_id,))
            rows = c.fetchall()

            for row in rows:
                if row[0] and os.path.exists(row[0]):
                    os.remove(row[0])

            c.execute('DELETE FROM snapshots WHERE instance_id = ?', (instance_id,))
            c.execute(f'DELETE FROM {VIOLATIONS_TABLE} WHERE instance_id = ?', (instance_id,))

            conn.commit()
            conn.close()

            return True
        except Exception as e:
            print(f"Error deleting violation record: {e}")
            return False
