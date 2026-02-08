import cv2
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ================= CONFIG =================
TRAIN_DIR = "train_good"
TEST_DIR  = "test_bad1"                 # changed to test_bad1
OUT_DIR   = "outputs2/ocsvm"            # changed to outputs2
CACHE_FILE = "train_sift_features.npy"

IMG_SIZE = (600, 600)

MAX_KEYPOINTS = 400
MAX_FEATURES = 5000

OCSVM_NU = 0.03
OCSVM_GAMMA = "scale"

MIN_AREA = 800
MAX_AREA = 30000

# Clean output folder before running
if os.path.exists(OUT_DIR):
    print("[INFO] Clearing old output folder...")
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
else:
    os.makedirs(OUT_DIR)

print("[INFO] Initializing SIFT...")

# ---------- ROI Mask ----------
def get_roi_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 50])
    upper = np.array([180, 255, 255])
    roi = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7,7), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    return roi

# ---------- Feature Extraction ----------
def extract_sift(img_gray):
    sift = cv2.SIFT_create(MAX_KEYPOINTS)
    kp, des = sift.detectAndCompute(img_gray, None)
    if des is None:
        return [], np.empty((0,128), np.float32)
    return kp, des.astype(np.float32)

# ---------- Load / Build Training Features ----------
if os.path.exists(CACHE_FILE):
    print("[INFO] Loading cached SIFT features...")
    X = np.load(CACHE_FILE, allow_pickle=True)
else:
    print("[INFO] Extracting SIFT features from training images...")
    all_desc = []

    for fname in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, fname)
        img = cv2.imread(path, 0)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        _, des = extract_sift(img)

        if len(des) > 0:
            all_desc.append(des)

    X = np.vstack(all_desc)

    if len(X) > MAX_FEATURES:
        idx = np.random.choice(len(X), MAX_FEATURES, replace=False)
        X = X[idx]

    np.save(CACHE_FILE, X)
    print(f"[INFO] Cached {len(X)} descriptors")

# ---------- Normalize ----------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------- Train OC-SVM ----------
print("[INFO] Training One-Class SVM...")
ocsvm = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma=OCSVM_GAMMA)
ocsvm.fit(X)
print("[INFO] OC-SVM training complete")

# ---------- Inference ----------
print("[INFO] Running anomaly localization...")

for fname in tqdm(sorted(os.listdir(TEST_DIR))):
    path = os.path.join(TEST_DIR, fname)

    img_color = cv2.imread(path)
    img_gray  = cv2.imread(path, 0)

    if img_color is None or img_gray is None:
        continue

    img_color = cv2.resize(img_color, IMG_SIZE)
    img_gray  = cv2.resize(img_gray, IMG_SIZE)

    roi = get_roi_mask(img_color)

    kp, des = extract_sift(img_gray)
    mask = np.zeros(img_gray.shape, dtype=np.uint8)

    if len(des) > 0:
        des = scaler.transform(des)
        preds = ocsvm.predict(des)

        for i, p in enumerate(preds):
            if p == -1:
                x, y = int(kp[i].pt[0]), int(kp[i].pt[1])
                cv2.circle(mask, (x, y), 12, 255, -1)

    # ---- Spread + smooth defect region
    kernel_big = np.ones((9,9), np.uint8)
    mask = cv2.dilate(mask, kernel_big, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

    # ---- Apply ROI
    mask = cv2.bitwise_and(mask, roi)

    # ---- Connected components filtering
    final_mask = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA < area < MAX_AREA:
            final_mask[labels == i] = 255

    # ---------- Save binary defect mask ----------
    cv2.imwrite(os.path.join(OUT_DIR, fname), final_mask)

    # ---------- Visualization (Red overlay) ----------
    overlay = img_color.copy()
    red_pixels = final_mask == 255
    overlay[red_pixels] = [0, 0, 255]   # Red in BGR

    alpha = 0.6
    visual = cv2.addWeighted(overlay, alpha, img_color, 1 - alpha, 0)

    vis_name = "visual_" + fname
    cv2.imwrite(os.path.join(OUT_DIR, vis_name), visual)

print("\n[DONE] SIFT + One-Class SVM AOI-style fault localization complete")
print(f"[OUTPUT] Results saved in folder: {OUT_DIR}")
