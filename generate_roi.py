# ---------- ONLY ROI GENERATION CODE ----------
# This will generate ROI images for all bad images and save them separately

import cv2
import os
import numpy as np
from tqdm import tqdm

TEST_DIR = "test_bad"
ROI_OUT_DIR = "outputs1/roi"
IMG_SIZE = (600, 600)

os.makedirs(ROI_OUT_DIR, exist_ok=True)

# ROI function (same as your code)
def get_roi_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 40, 50])
    upper = np.array([180, 255, 255])

    roi = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7,7), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    return roi


print("[INFO] Generating ROI masks for all bad images...")

for fname in tqdm(sorted(os.listdir(TEST_DIR))):
    path = os.path.join(TEST_DIR, fname)

    img_color = cv2.imread(path)
    if img_color is None:
        continue

    img_color = cv2.resize(img_color, IMG_SIZE)

    # Generate ROI
    roi = get_roi_mask(img_color)

    # Save ROI image
    save_path = os.path.join(ROI_OUT_DIR, fname)
    cv2.imwrite(save_path, roi)

print("[DONE] All ROI images are saved inside:", ROI_OUT_DIR)
