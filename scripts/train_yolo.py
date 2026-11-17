"""
train_yolo.py

Training/fine-tuning YOLOv8 untuk deteksi perilaku merokok, memakai
transfer learning dari model pretrained (COCO). Cocok dijalankan lokal
(kalau ada GPU) atau di Google Colab (paling praktis untuk mahasiswa).

Setup awal (sekali saja):
    pip install ultralytics

Contoh pakai:
    python train_yolo.py --data data.yaml --epochs 30 --model yolov8n.pt --imgsz 640

Setelah training selesai, hasil terbaik ada di:
    runs/detect/train/weights/best.pt
(salin file ini ke folder models/ untuk dipakai script inferensi nanti)

Catatan:
- yolov8n.pt = versi paling ringan/cepat, cocok untuk proof-of-concept awal.
- Kalau proof-of-concept sudah jalan dan mau akurasi lebih baik, coba
  yolov8s.pt atau yolov8m.pt (lebih besar, lebih akurat, lebih lambat).
"""

import argparse
import os
from ultralytics import YOLO


def train(data_yaml, model_name, epochs, imgsz, batch, patience, name, project):
    model = YOLO(model_name)
    return model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        patience=patience,
        project=os.path.abspath(project),
        name=name,
        exist_ok=False,
        workers=2,
    )


def main():
    parser = argparse.ArgumentParser(description="Training YOLOv8 untuk deteksi merokok")
    parser.add_argument("--data", default="data.yaml", help="Path ke file konfigurasi dataset (data.yaml)")
    parser.add_argument("--model", default="yolov8n.pt", help="Model pretrained awal (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch training")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar input")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (turunkan kalau GPU/Colab kena OOM)")
    parser.add_argument("--patience", type=int, default=30, help="Stop awal kalau gak membaik selama N epoch")
    parser.add_argument("--name", default="train", help="Nama run (pakai nama unik tiap training)")
    parser.add_argument("--project", default="runs_skripsi", help="Folder induk hasil training")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch,
          args.patience, args.name, args.project)


if __name__ == "__main__":
    main()