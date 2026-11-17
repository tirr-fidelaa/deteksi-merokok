"""dashboard/app.py - dashboard admin sistem deteksi merokok.
Jalankan dari folder root:  python dashboard/app.py
Buka di browser:            http://127.0.0.1:5000
"""
import hmac
import os
import re
import secrets
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlsplit

from dotenv import load_dotenv
from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_from_directory, session, url_for)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
load_dotenv(os.path.join(ROOT, ".env"))
import db

SNAP_DIR = os.path.join(ROOT, "snapshots")
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    raise SystemExit("ADMIN_PASSWORD belum diisi di .env")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
db.init_db()


# ---------- helper database ----------
def rows(sql, params=()):
    with closing(db.get_conn()) as conn:
        return conn.execute(sql, params).fetchall()


def one(sql, params=()):
    r = rows(sql, params)
    return r[0][0] if r else 0


# ---------- keamanan dasar ----------
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def cek_csrf():
    if request.method == "POST":
        sent = request.form.get("_csrf", "").encode()
        real = session.get("_csrf", "").encode()
        if not real or not hmac.compare_digest(sent, real):
            abort(400)


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


GAGAL = {}  # ip -> daftar waktu gagal login


def terlalu_banyak(ip):
    t = time.time()
    GAGAL[ip] = [x for x in GAGAL.get(ip, []) if t - x < 300]
    return len(GAGAL[ip]) >= 5


@app.template_filter("snap_url")
def snap_url(path):
    if not path:
        return ""
    rel = path.replace("\\", "/")
    if "snapshots/" in rel:
        rel = rel.split("snapshots/", 1)[1]
    return url_for("snapshot", filename=rel)


@app.route("/snapshot/<path:filename>")
@login_required
def snapshot(filename):
    return send_from_directory(SNAP_DIR, filename)


# ---------- login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = request.remote_addr
        if terlalu_banyak(ip):
            flash("Terlalu banyak percobaan. Coba lagi 5 menit lagi.", "danger")
        else:
            u = request.form.get("username", "").encode()
            p = request.form.get("password", "").encode()
            ok = (hmac.compare_digest(u, ADMIN_USER.encode())
                  and hmac.compare_digest(p, ADMIN_PASSWORD.encode()))
            if ok:
                session.clear()
                session["admin"] = True
                return redirect(url_for("beranda"))
            GAGAL.setdefault(ip, []).append(time.time())
            flash("Username atau password salah.", "danger")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Beranda ----------
@app.route("/")
@login_required
def beranda():
    now = datetime.now()
    hari_ini = now.strftime("%Y-%m-%d")
    seminggu = (now - timedelta(days=6)).strftime("%Y-%m-%d")
    return render_template(
        "beranda.html",
        total=one("SELECT COUNT(*) FROM kejadian"),
        n_hari_ini=one("SELECT COUNT(*) FROM kejadian WHERE date(waktu)=?", (hari_ini,)),
        n_minggu=one("SELECT COUNT(*) FROM kejadian WHERE date(waktu)>=?", (seminggu,)),
        terakhir=rows("SELECT waktu FROM kejadian ORDER BY id DESC LIMIT 1"),
        terbaru=rows("SELECT * FROM kejadian ORDER BY id DESC LIMIT 8"),
    )


# ---------- Riwayat ----------
@app.route("/riwayat")
@login_required
def riwayat():
    dari = request.args.get("dari", "").strip()
    sampai = request.args.get("sampai", "").strip()
    kelas = request.args.get("kelas", "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    per = 20

    where, params = [], []
    if dari:
        where.append("date(waktu) >= ?"); params.append(dari)
    if sampai:
        where.append("date(waktu) <= ?"); params.append(sampai)
    if kelas:
        where.append("kelas = ?"); params.append(kelas)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    total = one(f"SELECT COUNT(*) FROM kejadian {clause}", params)
    data = rows(f"SELECT * FROM kejadian {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [per, (page - 1) * per])
    return render_template(
        "riwayat.html", data=data, total=total, page=page,
        pages=max(1, -(-total // per)),
        dari=dari, sampai=sampai, kelas=kelas,
        daftar_kelas=[r[0] for r in rows("SELECT DISTINCT kelas FROM kejadian ORDER BY kelas")],
    )


# ---------- Statistik ----------
@app.route("/statistik")
@login_required
def statistik():
    hari = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(13, -1, -1)]
    cnt = {r["d"]: r["n"] for r in rows(
        "SELECT date(waktu) d, COUNT(*) n FROM kejadian WHERE date(waktu)>=? GROUP BY d", (hari[0],))}
    jam = {r["h"]: r["n"] for r in rows(
        "SELECT strftime('%H', waktu) h, COUNT(*) n FROM kejadian GROUP BY h")}
    return render_template(
        "statistik.html",
        label_hari=[d[8:] + "/" + d[5:7] for d in hari],
        harian=[cnt.get(d, 0) for d in hari],
        per_jam=[jam.get(f"{h:02d}", 0) for h in range(24)],
        per_kelas=rows("SELECT kelas, COUNT(*) n, ROUND(AVG(confidence), 2) rata "
                       "FROM kejadian GROUP BY kelas ORDER BY n DESC"),
    )


# ---------- Pengaturan ----------
def mask_rtsp(url):
    if not url:
        return ""
    try:
        u = urlsplit(url)
        return f"rtsp://***@{u.hostname}:{u.port or 554}{u.path}"
    except ValueError:
        return "(tersimpan)"


@app.route("/pengaturan", methods=["GET", "POST"])
@login_required
def pengaturan():
    if request.method == "POST":
        chat_id = request.form.get("telegram_chat_id", "").strip()
        conf = request.form.get("conf_threshold", "").strip()
        rtsp = request.form.get("rtsp_url", "").strip()

        if chat_id and not re.fullmatch(r"-?\d+", chat_id):
            flash("Chat ID harus berupa angka (boleh diawali minus).", "danger")
            return redirect(url_for("pengaturan"))
        try:
            conf_f = float(conf)
        except ValueError:
            conf_f = -1
        if not 0.05 <= conf_f <= 0.95:
            flash("Confidence threshold harus antara 0,05 dan 0,95.", "danger")
            return redirect(url_for("pengaturan"))
        if rtsp and not rtsp.lower().startswith("rtsp://"):
            flash("URL RTSP harus diawali rtsp://", "danger")
            return redirect(url_for("pengaturan"))

        db.set_setting("telegram_chat_id", chat_id)
        db.set_setting("conf_threshold", conf_f)
        if rtsp:  # kosong = pertahankan nilai lama
            db.set_setting("rtsp_url", rtsp)
        flash("Pengaturan disimpan.", "success")
        return redirect(url_for("pengaturan"))

    return render_template(
        "pengaturan.html",
        chat_id=db.get_setting("telegram_chat_id", os.environ.get("TELEGRAM_CHAT_ID", "")),
        conf=db.get_setting("conf_threshold", "0.25"),
        rtsp_masked=mask_rtsp(db.get_setting("rtsp_url", "")),
    )


if __name__ == "__main__":
    # 127.0.0.1 = hanya bisa dibuka dari laptop ini
    app.run(host="127.0.0.1", port=5000, debug=False)