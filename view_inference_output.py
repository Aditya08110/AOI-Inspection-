import cv2
import os
import numpy as np

# -------- CONFIG --------
TEST_IMAGE = "test_bad/bad_001.png"   # change if you want another image
IMG_SIZE = (600, 600)

# -------- Load trained files --------
CACHE_FILE = "train_sift_features.npy"

# -------- Load trained OC-SVM & scaler --------
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Load training features
X = np.load(CACHE_FILE, allow_pickle=True)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train OC-SVM again (same as main code)
OCSVM_NU = 0.03
OCSVM_GAMMA = "scale"
ocsvm = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma=OCSVM_GAMMA)
ocsvm.fit(X)

# -------- ROI Function --------
def get_roi_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 50])
    upper = np.array([180, 255, 255])
    roi = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7,7), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    return roi

# -------- SIFT Function --------
MAX_KEYPOINTS = 400

def extract_sift(img_gray):
    sift = cv2.SIFT_create(MAX_KEYPOINTS)
    kp, des = sift.detectAndCompute(img_gray, None)
    if des is None:
        return [], np.empty((0,128), np.float32)
    return kp, des.astype(np.float32)

# -------- MAIN --------
print("[INFO] Running inference on one image...")

# Load test image
img_color = cv2.imread(TEST_IMAGE)
img_gray  = cv2.imread(TEST_IMAGE, 0)

if img_color is None or img_gray is None:
    print("Image not found!")
    exit()

img_color = cv2.resize(img_color, IMG_SIZE)
img_gray  = cv2.resize(img_gray, IMG_SIZE)

# ROI
roi = get_roi_mask(img_color)

# SIFT
kp, des = extract_sift(img_gray)

# Create mask
mask = np.zeros(img_gray.shape, dtype=np.uint8)

if len(des) > 0:
    des = scaler.transform(des)
    preds = ocsvm.predict(des)

    for i, p in enumerate(preds):
        if p == -1:
            x, y = int(kp[i].pt[0]), int(kp[i].pt[1])
            cv2.circle(mask, (x, y), 12, 255, -1)

# Show outputs
cv2.imshow("Original Image", img_color)
cv2.imshow("ROI Mask", roi)
cv2.imshow("Defect Mask (Inference Output)", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save outputs
cv2.imwrite("roi_output.png", roi)
cv2.imwrite("defect_mask_output.png", mask)

print("[DONE] Outputs saved as:")
print(" - roi_output.png")
print(" - defect_mask_output.png")
