# Smoking Detection System — Skeleton Proyek

Struktur awal untuk skripsi "Deteksi Perilaku Merokok pada Video di Lingkungan Kampus
Menggunakan YOLOv8". Tahap ini fokus ke **Langkah 1 & 2**: ekstraksi frame dari video
dan proof-of-concept training YOLO. Dashboard, database, dan bot Telegram dibangun
di tahap berikutnya, setelah model terbukti jalan.

## Struktur folder

```
smoking_detection/
├── data/
│   ├── raw_videos/          # taruh video mentah hasil rekaman/dataset publik di sini
│   ├── frames/              # hasil ekstraksi frame otomatis (belum dilabeli)
│   └── dataset/             # dataset final SETELAH dilabeli, siap untuk training YOLO
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/   # anotasi format YOLO (.txt), 1 file per gambar
├── scripts/
│   ├── extract_frames.py    # ekstrak frame dari video di data/raw_videos
│   └── train_yolo.py        # training YOLOv8 (jalankan di Colab kalau tidak ada GPU lokal)
├── models/                  # tempat menyimpan hasil training (best.pt, dst.)
├── outputs/                 # hasil inferensi (video beranotasi, log CSV) — dipakai nanti
├── data.yaml                # konfigurasi dataset untuk YOLO
└── requirements.txt
```

## Alur kerja tahap ini

1. **Taruh video** di `data/raw_videos/` (boleh video kampus asli atau video contoh
   dari dataset publik untuk uji coba awal).
2. **Ekstrak frame**:
   ```bash
   python scripts/extract_frames.py --video data/raw_videos/contoh.mp4 --every-sec 1
   ```
   Ini akan menyimpan 1 gambar per detik ke `data/frames/contoh/`. Sesuaikan
   `--every-sec` sesuai kebutuhan (video panjang → interval lebih besar biar tidak
   kebanyakan gambar mirip-mirip).
3. **Labeli manual** gambar-gambar di `data/frames/` menggunakan
   [Roboflow](https://roboflow.com) atau [CVAT](https://www.cvat.ai/), lalu export
   dalam format **YOLO** (tiap gambar dapat pasangan file `.txt` berisi
   `class x_center y_center width height`, semua ternormalisasi 0–1).
4. **Pindahkan** hasil labeling ke `data/dataset/images/...` dan
   `data/dataset/labels/...` sesuai pembagian train/val/test (saran: 80/10/10).
5. **Training proof-of-concept**:
   ```bash
   python scripts/train_yolo.py --data data.yaml --epochs 30 --model yolov8n.pt
   ```
   Lebih enak dijalankan di **Google Colab** (ada GPU gratis) — tinggal upload folder
   `data/dataset/` + `data.yaml`, lalu jalankan isi `train_yolo.py` di cell Colab.

## Kelas objek (bisa diubah sesuai keputusan Anda)

Default di `data.yaml` ada 3 kelas: `rokok`, `asap`, `orang` (pendekatan
deteksi objek + konteks yang lebih robust dibanding cuma kelas `rokok` saja).
Ubah sesuai kesepakatan dengan dosen pembimbing kalau perlu.

## Catatan

- Mulai dengan dataset kecil (ratusan gambar) dulu untuk memastikan pipeline
  jalan dari ujung ke ujung, baru perbanyak data.
- Jangan lupa `.gitignore` video mentah & model besar kalau nanti proyek ini
  di-push ke GitHub (ukurannya bisa besar).
