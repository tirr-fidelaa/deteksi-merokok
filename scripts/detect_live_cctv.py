"""
detect_live_cctv.py

Deteksi merokok real-time dari stream RTSP (atau file video buat tes).
- Auto-reconnect kalau stream RTSP putus
- Selalu proses frame terbaru (frame lama dibuang)
- Kejadian valid: min-hits dalam window + cooldown
- Kejadian valid -> snapshot + insert SQLite + notifikasi Telegram

Contoh:
    # tes pakai file video
    python scripts/detect_live_cctv.py --url data/raw_videos/asset-smoker-6.mp4 --show

    # kamera asli (RTSP_URL diisi di .env)
    python scripts/detect_live_cctv.py --show

Jalankan dari folder root project. Stop: tekan 'q' di jendela video / Ctrl+C.
"""

import argparse
import os
import threading
import time
from collections import deque
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
from ultralytics import YOLO

import db
import notifier


class StreamReader:
    """Baca sumber video di thread terpisah, simpan cuma frame terbaru.
    RTSP: otomatis konek ulang kalau putus. File video: berhenti di akhir file."""

    def __init__(self, source: str, reconnect_delay: float = 3.0):
        self.source = source
        self.is_file = os.path.isfile(source)
        self.reconnect_delay = reconnect_delay
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.ended = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                if self.is_file:
                    print(f"[ERROR] Gagal membuka file video: {self.source}")
                    self.ended = True
                    return
                print(f"[WARN] Gagal konek ke kamera, coba lagi {self.reconnect_delay:.0f} detik lagi...")
                time.sleep(self.reconnect_delay)
                continue

            print("[INFO] Sumber video terbuka." if self.is_file else "[INFO] Terhubung ke kamera.")

            frame_delay = (1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0)) if self.is_file else 0.0

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                if frame_delay:
                    time.sleep(frame_delay)
            cap.release()

            if self.is_file:
                self.ended = True
                return

            print("[WARN] Stream putus, mencoba konek ulang...")
            with self.lock:
                self.frame = None
            time.sleep(self.reconnect_delay)

    def read(self):
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False


def on_event(cls_name, conf, snapshot_path, ts, notify=True):
    """Dipanggil tiap ada kejadian valid: simpan ke DB + kirim Telegram."""
    path = snapshot_path.replace("\\", "/")
    event_id = db.insert_kejadian(ts.strftime("%Y-%m-%d %H:%M:%S"), cls_name, round(conf, 3), path)
    print(f"[EVENT #{event_id}] {ts:%H:%M:%S} - '{cls_name}' (conf {conf:.2f}) -> {path}")

    if notify:
        caption = (f"Terdeteksi: {cls_name}\n"
                   f"Waktu: {ts:%d-%m-%Y %H:%M:%S}\n"
                   f"Confidence: {conf:.2f}")
        notifier.send_async(caption, snapshot_path)


def run_live(args):
    if not args.url:
        raise SystemExit("Sumber kosong. Isi RTSP_URL di .env atau pakai --url.")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model tidak ditemukan: {args.model}")
    if args.min_hits > args.window:
        raise SystemExit("--min-hits tidak boleh lebih besar dari --window")

    db.init_db()
    model = YOLO(args.model)
    class_names = model.names

    recent = {c: deque(maxlen=args.window) for c in class_names.values()}
    episode_active = False
    last_valid = 0.0
    next_notify_at = 0.0
    notif_times = deque()

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] conf={args.conf}, imgsz={args.imgsz}, min-hits={args.min_hits} dari window={args.window}")
    print(f"[INFO] episode-gap={args.episode_gap}s, notify-cooldown={args.notify_cooldown}s, max-per-hour={args.max_per_hour}")
    print(f"[INFO] Telegram: {'MATI (--no-telegram)' if args.no_telegram else 'aktif'}")

    reader = StreamReader(args.url).start()
    last_id = -1
    fps_count, fps_t0 = 0, time.time()

    try:
        while True:
            frame_id, frame = reader.read()
            if frame is None or frame_id == last_id:
                if reader.ended:
                    print("[INFO] Video selesai.")
                    break
                time.sleep(0.01)
                continue
            last_id = frame_id

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

            best_conf = {}
            for box in results.boxes:
                name = class_names[int(box.cls[0])]
                best_conf[name] = max(best_conf.get(name, 0.0), float(box.conf[0]))

            now = time.time()
            annotated = None

            valid_names = []
            for name in class_names.values():
                recent[name].append(1 if name in best_conf else 0)
                if name in best_conf and sum(recent[name]) >= args.min_hits:
                    valid_names.append(name)

            if valid_names:
                last_valid = now
                if not episode_active:
                    episode_active = True
                    name = max(valid_names, key=lambda n: best_conf[n])

                    annotated = results.plot()
                    ts = datetime.now()
                    day_dir = os.path.join(args.snap_dir, f"{ts:%Y-%m-%d}")
                    os.makedirs(day_dir, exist_ok=True)
                    snap_path = os.path.join(day_dir, f"{ts:%H%M%S}_{name}_{best_conf[name]:.2f}.jpg")
                    cv2.imwrite(snap_path, annotated)

                    while notif_times and now - notif_times[0] > 3600:
                        notif_times.popleft()
                    can_notify = (not args.no_telegram
                                  and now >= next_notify_at
                                  and len(notif_times) < args.max_per_hour)
                    if can_notify:
                        next_notify_at = now + args.notify_cooldown
                        notif_times.append(now)
                    elif not args.no_telegram:
                        print("[INFO] Notif Telegram ditahan (jeda / batas per jam).")

                    on_event(name, best_conf[name], snap_path, ts, notify=can_notify)

            elif episode_active and now - last_valid > args.episode_gap:
                episode_active = False
                print(f"[INFO] Episode selesai (tidak ada deteksi {args.episode_gap:.0f} detik).")

            if args.show:
                if annotated is None:
                    annotated = results.plot()
                cv2.imshow("Deteksi (tekan q untuk keluar)", cv2.resize(annotated, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            fps_count += 1
            if now - fps_t0 >= 10:
                print(f"[INFO] Kecepatan deteksi: {fps_count / (now - fps_t0):.1f} FPS")
                fps_count, fps_t0 = 0, now

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan (Ctrl+C).")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description="Deteksi merokok live (RTSP atau file video)")
    p.add_argument("--url", default=os.environ.get("RTSP_URL", ""), help="URL RTSP atau path file video (default: RTSP_URL di .env)")
    p.add_argument("--model", default="models/best_cctv_v2_s1280.pt", help="Path model .pt")
    p.add_argument("--conf", type=float, default=0.25, help="Ambang confidence")
    p.add_argument("--imgsz", type=int, default=1280, help="Ukuran input (samakan dengan training)")
    p.add_argument("--min-hits", type=int, default=3, help="Minimal deteksi dalam window")
    p.add_argument("--window", type=int, default=10, help="Panjang window frame terakhir")
    p.add_argument("--episode-gap", type=float, default=30.0, help="Detik tanpa deteksi valid sebelum episode dianggap selesai")
    p.add_argument("--notify-cooldown", type=float, default=60.0, help="Jeda minimal (detik) antar notif Telegram")
    p.add_argument("--max-per-hour", type=int, default=10, help="Batas maksimal notif Telegram per jam")
    p.add_argument("--snap-dir", default="snapshots", help="Folder simpan snapshot")
    p.add_argument("--show", action="store_true", help="Tampilkan jendela video")
    p.add_argument("--no-telegram", action="store_true", help="Jangan kirim Telegram (buat tes)")
    run_live(p.parse_args())


if __name__ == "__main__":
    main()