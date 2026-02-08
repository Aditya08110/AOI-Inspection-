import cv2
import os
import numpy as np

TRAIN_DIR = "train_good"
SIZE = (600, 600)

images = []

for f in sorted(os.listdir(TRAIN_DIR)):
    img = cv2.imread(os.path.join(TRAIN_DIR, f), 0)
    if img is None:
        continue
    img = cv2.resize(img, SIZE)
    img = img.astype(np.float32)
    images.append(img)

images = np.array(images)

# Mean and standard deviation model
ref_mean = np.mean(images, axis=0)
ref_std  = np.std(images, axis=0) + 1e-6   # avoid divide by zero

np.save("model_ref_mean.npy", ref_mean)
np.save("model_ref_std.npy", ref_std)

print("[DONE] Reference model created")
print("Mean shape:", ref_mean.shape)
print("Std shape :", ref_std.shape)
