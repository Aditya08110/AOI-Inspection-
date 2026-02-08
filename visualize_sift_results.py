import cv2
import os

IMG_DIR  = "test_bad"
MASK_DIR = "outputs1/final"
OUT_DIR  = "outputs1/visual"

os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(MASK_DIR):
    img = cv2.imread(os.path.join(IMG_DIR, f))
    mask = cv2.imread(os.path.join(MASK_DIR, f), 0)

    if img is None or mask is None:
        continue

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    overlay = img.copy()
    overlay[mask > 0] = (0,0,255)

    result = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.imwrite(os.path.join(OUT_DIR, f), result)

print("[DONE] AOI visualization created in outputs1/visual")
