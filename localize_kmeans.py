import cv2, os, numpy as np
from sklearn.cluster import KMeans

TRAIN_DIR = "train_good"
TEST_DIR = "test_bad"
OUT_DIR = "outputs1/kmeans"
SIZE = (600,600)

os.makedirs(OUT_DIR, exist_ok=True)

# Collect pixel samples
pixels = []
for f in os.listdir(TRAIN_DIR):
    img = cv2.imread(os.path.join(TRAIN_DIR,f),0)
    img = cv2.resize(img,SIZE)
    pixels.append(img.reshape(-1,1))

pixels = np.vstack(pixels)

kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)

for f in os.listdir(TEST_DIR):
    img = cv2.imread(os.path.join(TEST_DIR,f),0)
    img = cv2.resize(img,SIZE)
    labels = kmeans.predict(img.reshape(-1,1))
    mask = labels.reshape(SIZE)
    mask = (mask != mask[0,0]).astype(np.uint8)*255
    cv2.imwrite(os.path.join(OUT_DIR,f),mask)

print("[DONE] KMeans maps → outputs1/kmeans")
