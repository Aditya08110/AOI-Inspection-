import cv2
import numpy as np

img = cv2.imread("train_good/good_001.png")
img = cv2.resize(img, (600,600))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# mask solder + copper (not green PCB)
lower = np.array([0, 30, 60])
upper = np.array([180, 255, 255])
roi = cv2.inRange(hsv, lower, upper)

kernel = np.ones((7,7), np.uint8)
roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("roi_mask.png", roi)
print("ROI mask saved as roi_mask.png")
