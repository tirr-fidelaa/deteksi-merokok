"""
extract_frames.py

Mengekstrak frame dari sebuah video pada interval waktu tertentu, untuk
kemudian dilabeli manual (Roboflow/CVAT) sebagai dataset training YOLO.

Contoh pakai:
    python extract_frames.py --video data/raw_videos/contoh.mp4 --every-sec 1
    python extract_frames.py --video data/raw_videos/contoh.mp4 --every-sec 0.5 --out data/frames

Hasil disimpan di: data/frames/<nama_video>/frame_00001.jpg, frame_00002.jpg, dst.
"""

import argparse
import os
import cv2


def extract_frames(video_path: str, out_dir: str, every_sec: float) -> int:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(out_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_interval = max(1, round(fps * every_sec))

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            saved_idx += 1
            filename = os.path.join(save_dir, f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(filename, frame)

        frame_idx += 1

    cap.release()
    print(f"[OK] {saved_idx} frame disimpan ke: {save_dir}")
    print(f"     (fps terdeteksi ~{fps:.1f}, interval {frame_interval} frame / {every_sec}s)")
    return saved_idx


def main():
    parser = argparse.ArgumentParser(description="Ekstrak frame dari video untuk dataset YOLO")
    parser.add_argument("--video", required=True, help="Path ke file video sumber")
    parser.add_argument("--out", default="data/frames", help="Folder output hasil ekstraksi")
    parser.add_argument(
        "--every-sec",
        type=float,
        default=1.0,
        help="Ambil 1 frame setiap N detik (default: 1.0). Video panjang → pakai angka lebih besar.",
    )
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.every_sec)


if __name__ == "__main__":
    main()
