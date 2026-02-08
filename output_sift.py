import cv2
import numpy as np

# ---------------- CONFIG ----------------
MAX_KEYPOINTS = 400
IMG_SIZE = (600, 600)

# Use any one of your real test images
IMAGE_PATH = "test_bad/bad_001.png"   # change if you want another image

# ---------------- SIFT FUNCTION ----------------
def extract_sift(img_gray):
    sift = cv2.SIFT_create(MAX_KEYPOINTS)
    kp, des = sift.detectAndCompute(img_gray, None)
    if des is None:
        return [], np.empty((0,128), np.float32)
    return kp, des.astype(np.float32)

# ---------------- MAIN ----------------
# Load image in grayscale
img = cv2.imread(IMAGE_PATH, 0)

if img is None:
    print("Image not found. Check file name and path.")
    exit()

# Resize image
img = cv2.resize(img, IMG_SIZE)

# Extract SIFT features
kp, des = extract_sift(img)

print("Image used:", IMAGE_PATH)
print("Number of keypoints:", len(kp))
print("Descriptor shape:", des.shape)

# Draw SIFT keypoints on the image
img_kp = cv2.drawKeypoints(
    img, kp, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Show result
cv2.imshow("SIFT Keypoints Output", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output image
cv2.imwrite("sift_keypoints_output.png", img_kp)
print("SIFT visualization saved as sift_keypoints_output.png")
