import cv2
import os

OCSVM_DIR = "outputs1/ocsvm"
FINAL_DIR = "outputs1/final"
os.makedirs(FINAL_DIR, exist_ok=True)

print("[INFO] Combining maps...")

for f in os.listdir(OCSVM_DIR):
    src = os.path.join(OCSVM_DIR, f)
    img = cv2.imread(src, 0)
    if img is None: continue
    cv2.imwrite(os.path.join(FINAL_DIR, f), img)

print("[DONE] Final maps saved in outputs1/final")
