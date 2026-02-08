import cv2
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ================= CONFIG =================
TRAIN_DIR = "train_good"
TEST_DIR  = "test_bad"
OUT_DIR   = "outputs1/ocsvm" #
CACHE_FILE = "train_sift_features.npy" # Cache file for SIFT features

IMG_SIZE = (600, 600) # Resize images to this size

MAX_KEYPOINTS = 400 # Max SIFT keypoints per image
MAX_FEATURES = 5000 # Max descriptors to use for training

OCSVM_NU = 0.03 # Anomaly fraction
OCSVM_GAMMA = "scale" # RBF kernel parameter

# Area filtering for final blobs
MIN_AREA = 800  # Minimum area to consider a defect  
MAX_AREA = 30000 # Create output directory if not exists

os.makedirs(OUT_DIR, exist_ok=True) # Create output directory if not exists

# =========================================

print("[INFO] Initializing SIFT...")
# ---------- ROI Mask (Solder + Copper, not green PCB) ----------
def get_roi_mask(img_bgr):
    # Convert to HSV and threshold for non-green areas
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Define range for non-green colors
    lower = np.array([0, 40, 50])
    upper = np.array([180, 255, 255])# Create mask for non-green areas 
    roi = cv2.inRange(hsv, lower, upper)# Morphological closing to fill small holes

    kernel = np.ones((7,7), np.uint8)# Apply morphological closing
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)# Final ROI mask
    return roi# Initialize SIFT detector 


# ---------- Feature Extraction ----------
def extract_sift(img_gray):# Initialize SIFT detector
    sift = cv2.SIFT_create(MAX_KEYPOINTS)# Detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(img_gray, None)# Handle case with no descriptors
    if des is None:# Return empty arrays if no descriptors found
        return [], np.empty((0,128), np.float32)# Return keypoints and descriptors
    return kp, des.astype(np.float32)# ---------- Main Pipeline ----------


# ---------- Load / Build Training Features ----------
if os.path.exists(CACHE_FILE):# Load cached features
    print("[INFO] Loading cached SIFT features...")# Load features from .npy file
    X = np.load(CACHE_FILE, allow_pickle=True)# Print number of loaded descriptors
else:
    print("[INFO] Extracting SIFT features from training images...")# Initialize list to hold all descriptors
    all_desc = []# Loop over training images

    for fname in tqdm(os.listdir(TRAIN_DIR)):# Read and preprocess image
        path = os.path.join(TRAIN_DIR, fname)#  Load image in grayscale
        img = cv2.imread(path, 0)# Check if image is loaded successfully
        if img is None:# Skip if image loading failed
            continue# Resize image to standard size
        img = cv2.resize(img, IMG_SIZE)# Extract SIFT keypoints and descriptors

        _, des = extract_sift(img)# Append descriptors if available
        if len(des) > 0:# Add descriptors to list
            all_desc.append(des)# Stack all descriptors into a single array

    X = np.vstack(all_desc)# Print total number of descriptors extracted

    if len(X) > MAX_FEATURES:# Subsample descriptors if exceeding MAX_FEATURES
        idx = np.random.choice(len(X), MAX_FEATURES, replace=False)# Select random indices
        X = X[idx]# Subsampled descriptors

    np.save(CACHE_FILE, X)# Cache descriptors to .npy file
    print(f"[INFO] Cached {len(X)} descriptors")# ---------- Feature Normalization ----------

# ---------- Normalize ----------
scaler = StandardScaler()# Fit scaler and transform features
X = scaler.fit_transform(X)# Print completion of normalization

# ---------- Train OC-SVM ----------
print("[INFO] Training One-Class SVM...")# Initialize and train OC-SVM model
ocsvm = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma=OCSVM_GAMMA)#  Fit model to training data
ocsvm.fit(X)# Print completion of training  
print("[INFO] OC-SVM training complete")# ----------

# ---------- Inference ----------
print("[INFO] Running anomaly localization...")# Loop over test images

for fname in tqdm(sorted(os.listdir(TEST_DIR))):# Read and preprocess image
    path = os.path.join(TEST_DIR, fname)#  Load color and grayscale images

    img_color = cv2.imread(path)# Load grayscale image
    img_gray  = cv2.imread(path, 0)# Check if images are loaded successfully

    if img_color is None or img_gray is None:# Skip if image loading failed
        continue# Resize images to standard size

    img_color = cv2.resize(img_color, IMG_SIZE)# Resize grayscale image
    img_gray  = cv2.resize(img_gray, IMG_SIZE)# Extract ROI mask

    roi = get_roi_mask(img_color)# Extract SIFT keypoints and descriptors

    kp, des = extract_sift(img_gray)# Initialize empty mask for defects
    mask = np.zeros(img_gray.shape, dtype=np.uint8)# Predict anomalies if descriptors are available

    if len(des) > 0:# Normalize descriptors and predict with OC-SVM
        des = scaler.transform(des) # Predict using trained OC-SVM
        preds = ocsvm.predict(des) # Mark keypoints predicted as anomalies in the mask

        for i, p in enumerate(preds):# If predicted as anomaly
            if p == -1: # Get keypoint coordinates
                x, y = int(kp[i].pt[0]), int(kp[i].pt[1]) # Draw circle on mask
                cv2.circle(mask, (x, y), 12, 255, -1) # Morphological operations to refine mask

    # ---- Spread + smooth defect region
    kernel_big = np.ones((9,9), np.uint8) # Dilate and close to fill gaps
    mask = cv2.dilate(mask, kernel_big, iterations=2) # Close gaps in mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big) # Gaussian blur to smooth edges

    # ---- Apply ROI constraint (MOST IMPORTANT)
    mask = cv2.bitwise_and(mask, roi) # Mask with ROI to keep only relevant areas

    # ---- Remove noise using connected components
    final_mask = np.zeros_like(mask) # Connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8) # Filter components by area

    for i in range(1, num_labels): # Skip background
        area = stats[i, cv2.CC_STAT_AREA] # Keep component if area is within thresholds
        if MIN_AREA < area < MAX_AREA: # Add to final mask
            final_mask[labels == i] = 255 # Save final mask to output directory

    cv2.imwrite(os.path.join(OUT_DIR, fname), final_mask) # Print completion of inference

print("\n[DONE] SIFT + One-Class SVM AOI-style fault localization complete")
print(f"[OUTPUT] Results saved in folder: {OUT_DIR}") 
