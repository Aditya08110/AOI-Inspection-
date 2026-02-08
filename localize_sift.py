import cv2
import os
import numpy as np

TRAIN_DIR = "train_good"
TEST_DIR  = "test_bad"
OUT_DIR   = "outputs1/sift"
SIZE = (600,600)

os.makedirs(OUT_DIR, exist_ok=True)

sift = cv2.SIFT_create()

# ---- Build reference descriptors from all good samples ----
ref_descs = []

for f in os.listdir(TRAIN_DIR):
    img = cv2.imread(os.path.join(TRAIN_DIR, f), 0)
    if img is None:
        continue
    img = cv2.resize(img, SIZE)
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        ref_descs.append(des)

ref_descs = np.vstack(ref_descs)

# FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ---- Process test images ----
for fname in sorted(os.listdir(TEST_DIR)):
    img = cv2.imread(os.path.join(TEST_DIR, fname), 0)
    if img is None:
        continue
    img = cv2.resize(img, SIZE)

    kp, des = sift.detectAndCompute(img, None)
    mask = np.zeros(SIZE, np.uint8)

    if des is not None:
        matches = flann.knnMatch(des, ref_descs, k=2)

        for m,n in matches:
            if m.distance > 0.7 * n.distance:
                x,y = kp[m.queryIdx].pt
                cv2.circle(mask, (int(x),int(y)), 6, 255, -1)

    mask = cv2.dilate(mask, np.ones((7,7),np.uint8), iterations=1)
    cv2.imwrite(os.path.join(OUT_DIR, fname), mask)

print("[DONE] SIFT anomaly maps created → outputs1/sift")
