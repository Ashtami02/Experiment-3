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
orb = cv2.ORB_create(nfeatures=100,      
        scaleFactor=2,
        nlevels=3,          
        edgeThreshold=100,
        fastThreshold=20)
kp1, des1 = orb.detectAndCompute(img1, None)
if img1 is None or img2 is None:
    print("Could not open or find the images.")
    exit()
percent_kept = []   # e.g., 100.00, 99.75, 99.50, ...
scores = []
    
for i in range(150):
    percent = 100 - i * 0.5      
    scale = percent / 100.0                        

    new_h = int(h * scale)
    new_w = int(w * scale)

    y1 = h//2 - new_h//2
    y2 = h//2 + new_h//2
    x1 = w//2 - new_w//2
    x2 = w//2 + new_w//2

    crop = img2[y1:y2, x1:x2]

    kp2, des2 = orb.detectAndCompute(crop, None)

    end_orb = datetime.now()
    print(f"ORB detection time: {(end_orb - start_orb).total_seconds():.4f} seconds")

    if des1 is None or des2 is None:
        print("Not enough descriptors.")
        score=0
        percent_kept.append(percent)
        scores.append(score)
        continue  # go to next iteration
    start_knn = datetime.now()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    ratio_thresh = 0.75
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    end_knn = datetime.now()
    print(f"KNN + Ratio test time: {(end_knn - start_knn).total_seconds():.4f} seconds")
    print(f"Good matches after ratio test: {len(good_matches)}")

    start_ransac = datetime.now()

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99
    )

    end_ransac = datetime.now()
    print(f"RANSAC time: {(end_ransac - start_ransac).total_seconds():.4f} seconds")

    if mask is None or mask.size == 0:
        print("RANSAC failed or returned empty mask.")
        inliers = 0
        score = 0
        percent_kept.append(percent)
        scores.append(score)
        continue  # go to next iteration

# ---------- NORMAL CASE ----------
    inliers = int(mask.sum())
    total_good_matches = len(good_matches)
    score = inliers / total_good_matches if total_good_matches > 0 else 0


    print(f"Total good matches : {total_good_matches}")
    print(f"RANSAC inliers     : {inliers}")
    print(f"Matching score    : {score:.3f}")

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

    img_inliers = cv2.drawMatches(
        img1, kp1, crop, kp2,
        inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    percent_kept.append(percent)  # e.g., 100.00, 99.75, 99.50, ...
    scores.append(score)
    print(i)

plt.plot(percent_kept, scores, marker='o')
plt.xlabel("Percent of image kept")
plt.gca().invert_xaxis()
plt.ylabel("Matching score")
plt.title("Matching score vs image crop")
plt.grid(True)
plt.show()
    
