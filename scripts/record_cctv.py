import os
import time
import cv2
from datetime import datetime

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

RTSP_URL = "rtsp://admin:PASSWORDMU@192.168.1.xxx:554/tenda"
OUTPUT_DIR = "rekaman_cctv"
SEGMENT_MENIT = 5
FPS_REKAM = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("GAGAL membuka stream.")
    raise SystemExit

lebar = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi: {lebar}x{tinggi}. Tekan 'q' untuk berhenti.")

def buka_writer():
    nama = datetime.now().strftime("cctv_%Y%m%d_%H%M%S.mp4")
    path = os.path.join(OUTPUT_DIR, nama)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    print("Mulai rekam:", path)
    return cv2.VideoWriter(path, fourcc, FPS_REKAM, (lebar, tinggi))

writer = buka_writer()
waktu_mulai = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame gagal dibaca, berhenti.")
        break

    writer.write(frame)

    cv2.imshow("Merekam (q = stop)", cv2.resize(frame, (960, 540)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if time.time() - waktu_mulai >= SEGMENT_MENIT * 60:
        writer.release()
        writer = buka_writer()
        waktu_mulai = time.time()

writer.release()
cap.release()
cv2.destroyAllWindows()
print("Selesai.")