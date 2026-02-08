import cv2
import os
import numpy as np

REF_MEAN = "model_ref_mean.npy"
REF_STD  = "model_ref_std.npy"
TEST_DIR = "test_bad"
OUT_DIR  = "outputs/mask"

os.makedirs(OUT_DIR, exist_ok=True)

SIZE = (600,600)
Z_THRESHOLD = 4.0      # how strict the detector is
MIN_AREA = 200
MAX_AREA = 10000

# Load reference
ref_mean = np.load(REF_MEAN)
ref_std  = np.load(REF_STD)

# Normalize reference
ref_mean = ref_mean.astype(np.float32)
ref_std  = ref_std.astype(np.float32)

# Lighting normalization
clahe = cv2.createCLAHE(2.0, (8,8))

for fname in os.listdir(TEST_DIR):
    img = cv2.imread(os.path.join(TEST_DIR, fname), 0)
    if img is None:
        continue

    img = cv2.resize(img, SIZE)
    img = img.astype(np.float32)
    img = clahe.apply(img.astype(np.uint8)).astype(np.float32)

    # Z-score anomaly map
    z_map = np.abs(img - ref_mean) / ref_std

    # Binary defect mask
    mask = (z_map > Z_THRESHOLD).astype(np.uint8) * 255

    # Morphology cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Blob filtering
    final = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA < area < MAX_AREA:
            final[labels == i] = 255

    cv2.imwrite(os.path.join(OUT_DIR, fname), final)

print("[DONE] Unified AOI fault localization complete")
