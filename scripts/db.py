"""db.py - akses SQLite untuk sistem deteksi merokok."""
import os
import sqlite3
from contextlib import closing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "smoker.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with closing(get_conn()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kejadian (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                waktu         TEXT NOT NULL,
                kelas         TEXT NOT NULL,
                confidence    REAL NOT NULL,
                path_snapshot TEXT,
                path_klip     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pengaturan (
                kunci TEXT PRIMARY KEY,
                nilai TEXT NOT NULL
            )
        """)


def insert_kejadian(waktu, kelas, confidence, path_snapshot, path_klip=None):
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO kejadian (waktu, kelas, confidence, path_snapshot, path_klip) "
            "VALUES (?, ?, ?, ?, ?)",
            (waktu, kelas, confidence, path_snapshot, path_klip),
        )
        return cur.lastrowid


def get_setting(kunci, default=None):
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT nilai FROM pengaturan WHERE kunci = ?", (kunci,)).fetchone()
    return row["nilai"] if row else default


def set_setting(kunci, nilai):
    with closing(get_conn()) as conn, conn:
        conn.execute(
            "INSERT INTO pengaturan (kunci, nilai) VALUES (?, ?) "
            "ON CONFLICT(kunci) DO UPDATE SET nilai = excluded.nilai",
            (kunci, str(nilai)),
        )