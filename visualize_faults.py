import cv2, os
import numpy as np

IMG_DIR = "test_bad"
MASK_DIR =  "outputs/mask"
OUT_DIR = "outputs/visual"

os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, f)
    mask_path = os.path.join(MASK_DIR, f)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        continue

    # 🔑 Resize mask to match image size
    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img.copy()
    overlay[mask > 0] = (0, 0, 255)

    result = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    cv2.imwrite(os.path.join(OUT_DIR, f), result)

print("[DONE] Visualization complete")
