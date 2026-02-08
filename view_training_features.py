import numpy as np

CACHE_FILE = "train_sift_features.npy"

X = np.load(CACHE_FILE, allow_pickle=True)

print("Training feature matrix shape:", X.shape)
print("First 5 feature vectors:\n", X[:5])
