import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 

start_orb = datetime.now()

imgg = cv2.imread('yes.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('yes2.jpg', cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(imgg, (640, 480))
img2 = cv2.resize(img, (640, 480))

h, w = img2.shape[:2]

orb = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    fastThreshold=20
)

kp1, des1 = orb.detectAndCompute(img1, None)

if img1 is None or img2 is None:
    print("Could not open or find the images.")
    exit()

percent_kept = []
scores = []
num_matches_list = []
num_inliers_list = []

for i in range(146):
    percent = 100 - i * 0.5
    scale = percent / 100.0

    new_h = int(h * scale)
    new_w = int(w * scale)
    remove_h = int((h - new_h) / 2)
    remove_w = int((w - new_w) / 2)

    y1 = remove_h
    y2 = h - remove_h
    x1 = remove_w
    x2 = w - remove_w
    crop = img2[y1:y2, x1:x2]

    kp2, des2 = orb.detectAndCompute(crop, None)

    if des2 is None or des1 is None:
        percent_kept.append(percent)
        scores.append(0)
        num_matches_list.append(0)
        num_inliers_list.append(0)
        print(f"{i} -- Not enough descriptors")
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_thresh = 0.75
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    total_good_matches = len(good_matches)

    if total_good_matches < 8:
        percent_kept.append(percent)
        scores.append(0)
        num_matches_list.append(total_good_matches)
        num_inliers_list.append(0)
        print(f"{i} -- Not enough good matches")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99
    )

    if mask is None:
        inliers = 0
    else:
        inliers = int(mask.sum())

    score = inliers / total_good_matches if total_good_matches > 0 else 0

    print(f"Percent kept: {percent:.2f}, Matches={total_good_matches}, Inliers={inliers}, Score={score:.3f}")

    percent_kept.append(percent)
    scores.append(score)
    num_matches_list.append(total_good_matches)
    num_inliers_list.append(inliers)

plt.figure()
plt.plot(percent_kept, num_matches_list)
plt.xlabel("Percent of image kept")
plt.ylabel("Number of good matches")
plt.title("Good matches vs crop percentage in orb")
plt.grid(True)
plt.gca().invert_xaxis()

plt.figure()
plt.plot(percent_kept, scores)
plt.gca().invert_xaxis()
plt.xlabel("Percent of image kept")
plt.ylabel("RANSAC score (inliers / good matches)")
plt.title("RANSAC score vs crop percentage")
plt.grid(True)
plt.show()


