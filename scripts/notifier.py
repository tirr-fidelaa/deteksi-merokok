"""notifier.py - kirim notifikasi ke grup Telegram.

Tes kirim pesan:  python scripts/notifier.py --test
"""
import os
import sys
import threading

import requests
from dotenv import load_dotenv

load_dotenv()
import db


def _get_config():
    db.init_db()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (db.get_setting("telegram_chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
    return token, chat_id


def _send(caption, photo_path=None):
    token, chat_id = _get_config()
    if not token or not chat_id:
        print("[WARN] Telegram belum diatur (token/chat ID kosong), notifikasi dilewati.")
        return False

    base = f"https://api.telegram.org/bot{token}"
    try:
        if photo_path and os.path.isfile(photo_path):
            with open(photo_path, "rb") as f:
                r = requests.post(f"{base}/sendPhoto",
                                  data={"chat_id": chat_id, "caption": caption},
                                  files={"photo": f}, timeout=20)
        else:
            r = requests.post(f"{base}/sendMessage",
                              data={"chat_id": chat_id, "text": caption}, timeout=20)
        if r.ok:
            return True
        print(f"[WARN] Telegram menolak: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[WARN] Telegram gagal terkirim ({type(e).__name__})")
    return False


def send_async(caption, photo_path=None):
    """Kirim di thread terpisah supaya deteksi gak ketahan kalau internet lambat."""
    threading.Thread(target=_send, args=(caption, photo_path), daemon=True).start()


if __name__ == "__main__":
    if "--test" in sys.argv:
        ok = _send("Tes notifikasi sistem deteksi merokok")
        print("[OK] Terkirim." if ok else "[GAGAL] Cek token dan chat ID di .env")