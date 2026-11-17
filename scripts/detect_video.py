"""
detect_video.py

Menjalankan model YOLO hasil training pada sebuah video utuh.
Menghasilkan:
  1. Video baru dengan bounding box hasil deteksi digambar di tiap frame
  2. File log CSV berisi tiap kejadian terdeteksi (waktu, kelas, confidence)

Contoh pakai:
    python scripts/detect_video.py --video data/raw_videos/asset-smoker-6.mp4 \
        --model models/best.pt --conf 0.25 --imgsz 640 --min-hits 3 --window 10

Catatan: --imgsz sebaiknya SAMA dengan imgsz saat training model tsb.
"""

import argparse
import csv
import os
from collections import deque

import cv2
from ultralytics import YOLO


def format_timestamp(seconds: float) -> str:
    """Ubah detik jadi format mm:ss biar gampang dibaca di log/laporan."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def run_detection(
    video_path: str,
    model_path: str,
    conf_threshold: float,
    imgsz: int,
    min_hits: int,
    window: int,
    cooldown: float,
    out_dir: str,
):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if min_hits > window:
        raise ValueError(f"--min-hits ({min_hits}) tidak boleh lebih besar dari --window ({window})")

    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    model = YOLO(model_path)
    class_names = model.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(out_dir, f"{video_name}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    out_log_path = os.path.join(out_dir, f"{video_name}_log.csv")
    log_file = open(out_log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "frame_number", "class", "confidence"])

    recent = {c: deque(maxlen=window) for c in class_names.values()}
    cooldown_until = {c: -1.0 for c in class_names.values()}

    frame_idx = 0
    total_valid_events = 0

    print(f"[INFO] Memproses video: {video_path}")
    print(f"[INFO] FPS: {fps:.1f}, Resolusi: {width}x{height}")
    print(f"[INFO] conf={conf_threshold}, imgsz={imgsz}, min-hits={min_hits} dari window={window}, cooldown={cooldown}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        results = model.predict(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)[0]

        best_conf = {}

        for box in results.boxes:
            cls_name = class_names[int(box.cls[0])]
            confidence = float(box.conf[0])
            best_conf[cls_name] = max(best_conf.get(cls_name, 0.0), confidence)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{cls_name} {confidence:.2f}"
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        for cls_name in class_names.values():
            recent[cls_name].append(1 if cls_name in best_conf else 0)

            if (
                cls_name in best_conf
                and sum(recent[cls_name]) >= min_hits
                and timestamp_sec >= cooldown_until[cls_name]
            ):
                total_valid_events += 1
                cooldown_until[cls_name] = timestamp_sec + cooldown
                log_writer.writerow([
                    format_timestamp(timestamp_sec),
                    frame_idx,
                    cls_name,
                    f"{best_conf[cls_name]:.2f}",
                ])
                print(f"[EVENT] {format_timestamp(timestamp_sec)} - '{cls_name}' "
                      f"terdeteksi (conf {best_conf[cls_name]:.2f})")

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    log_file.close()

    print(f"\n[OK] Selesai. Total kejadian valid terdeteksi: {total_valid_events}")
    print(f"     Video hasil   : {out_video_path}")
    print(f"     Log kejadian  : {out_log_path}")

    return out_video_path, out_log_path, total_valid_events


def main():
    parser = argparse.ArgumentParser(description="Deteksi perilaku merokok pada video menggunakan model YOLO")
    parser.add_argument("--video", required=True, help="Path ke video yang mau dideteksi")
    parser.add_argument("--model", default="models/best.pt", help="Path ke model hasil training (best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Ambang batas confidence (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran input inferensi, samakan dengan imgsz training")
    parser.add_argument("--min-hits", type=int, default=3, help="Minimal jumlah frame terdeteksi dalam window (default: 3)")
    parser.add_argument("--window", type=int, default=10, help="Panjang window frame terakhir (default: 10)")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Jeda (detik) sebelum kelas yang sama dicatat lagi (default: 10)")
    parser.add_argument("--out", default="outputs", help="Folder untuk menyimpan video hasil & log")
    args = parser.parse_args()

    run_detection(
        args.video, args.model, args.conf, args.imgsz,
        args.min_hits, args.window, args.cooldown, args.out,
    )


if __name__ == "__main__":
    main()