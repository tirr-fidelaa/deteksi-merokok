import sys
from ultralytics import YOLO

MODEL_PATH = "models/best_cctv_v2_s1280.pt"
DATA_YAML = "data.yaml"
IMGSZ = None

model = YOLO(MODEL_PATH)

print("=" * 50)
print("INFO TRAINING (dari dalam file model)")
print("=" * 50)
args = model.ckpt.get("train_args", {})
print("imgsz   :", args.get("imgsz"))
print("data    :", args.get("data"))
print("epochs  :", args.get("epochs"))
print("batch   :", args.get("batch"))
print("model   :", args.get("model"))
print("name    :", args.get("name"))
print("tanggal :", model.ckpt.get("date"))
print("kelas   :", model.names)

imgsz = IMGSZ or args.get("imgsz") or 640

for split in ["val", "test"]:
    print()
    print("=" * 50)
    print(f"HASIL EVALUASI di split '{split}' (imgsz={imgsz})")
    print("=" * 50)
    try:
        m = model.val(data=DATA_YAML, split=split, imgsz=imgsz,
                      workers=0, batch=4, verbose=False, plots=False)
    except Exception as e:
        print(f"Split '{split}' gagal dievaluasi: {e}")
        continue

    print(f"{'kelas':<12} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    for i, cls_id in enumerate(m.box.ap_class_index):
        nama = model.names[int(cls_id)]
        p, r, ap50, ap = m.box.class_result(i)
        print(f"{nama:<12} {p:>7.3f} {r:>7.3f} {ap50:>7.3f} {ap:>9.3f}")
    print(f"{'SEMUA':<12} {m.box.mp:>7.3f} {m.box.mr:>7.3f} {m.box.map50:>7.3f} {m.box.map:>9.3f}")