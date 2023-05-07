import cv2
import numpy as np
import matplotlib.pyplot as plt

I1 = cv2.imread('./data/myL.jpeg');
I2 = cv2.imread('./data/myR.jpeg');

I1gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I2gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(I1gray, None)
kp2, des2 = orb.detectAndCompute(I2gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)[:30]

img3 = cv2.drawMatches(I1,kp1,I2,kp2, matches, flags=2, outImg = None)
plt.imshow(img3), plt.show()

best_kp1 = []
best_kp2 = []
best_matches = []

for match in matches:
	best_kp1.append(kp1[match.queryIdx].pt)
	best_kp2.append(kp2[match.trainIdx].pt)
	best_matches.append(match)

best_kp1 = np.array(best_kp1)
best_kp2 = np.array(best_kp2)
best_matches = np.array(best_matches)

F, inlier_mask = cv2.findFundamentalMat(best_kp1, best_kp2, cv2.FM_7POINT)
inlier_mask = inlier_mask.flatten()

#points within epipolar lines
inlier_kp1 = best_kp1[inlier_mask == 1]
inlier_kp2 = best_kp2[inlier_mask == 1]

inlier_matches = best_matches[inlier_mask==1]


img3 = cv2.drawMatches(I1,kp1,I2,kp2, inlier_matches, flags=2, outImg = None)
plt.imshow(img3),plt.show()

thresh = 0

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(inlier_kp1), np.float32(inlier_kp2), F, I1gray.shape[::-1], 1)

I1_rect = np.float32([[[0, 0], [I1.shape[1], 0], [I1.shape[1], I1.shape[0]], [0, I1.shape[0]]]])
warped_I1_rect = cv2.perspectiveTransform(I1_rect, H1)

I2_rect = np.float32([[[0, 0], [I2.shape[1], 0], [I2.shape[1], I2.shape[0]], [0, I2.shape[0]]]])
warped_I2_rect = cv2.perspectiveTransform(I2_rect, H2)

min_x_I1= min(warped_I1_rect[0][0][0], warped_I1_rect[0][1][0], warped_I1_rect[0][2][0], warped_I1_rect[0][3][0])
min_x_I2 =min(warped_I2_rect[0][0][0], warped_I2_rect[0][1][0], warped_I2_rect[0][2][0], warped_I2_rect[0][3][0])

min_y_I1 = min(warped_I1_rect[0][0][1], warped_I1_rect[0][1][1], warped_I1_rect[0][2][1], warped_I1_rect[0][3][1])
min_y_I2 = min(warped_I2_rect[0][0][1], warped_I2_rect[0][1][1], warped_I2_rect[0][2][1], warped_I2_rect[0][3][1])

max_x_I1 = max(warped_I1_rect[0][0][0], warped_I1_rect[0][1][0], warped_I1_rect[0][2][0], warped_I1_rect[0][3][0])
max_x_I2 = max(warped_I2_rect[0][0][0], warped_I2_rect[0][1][0], warped_I2_rect[0][2][0], warped_I2_rect[0][3][0])

max_y_I1 = max(warped_I1_rect[0][0][1], warped_I1_rect[0][1][1], warped_I1_rect[0][2][1], warped_I1_rect[0][3][1])
max_y_I2 = max(warped_I2_rect[0][0][1], warped_I2_rect[0][1][1], warped_I2_rect[0][2][1], warped_I2_rect[0][3][1])
 
translation_xy_I1 = np.array([max(0, -min_x_I1), max(0, -min_y_I1)])
translation_xy_I2 = np.array([max(0, -min_x_I2), max(0, -min_y_I2)])

W_I1 = (max_x_I1 + translation_xy_I1[0])
H_I1 = (max_y_I1 + translation_xy_I1[1])

W_I2 = (max_x_I2 + translation_xy_I2[0])
H_I2 = (max_y_I2 + translation_xy_I2[1])

transform_T = np.eye(3)
transform_T[0,2] = translation_xy_I1[0]
transform_T[1,2] = translation_xy_I1[1]
transform_T = transform_T[:2, :]

H1 = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ H1

transform_T = np.eye(3)
transform_T[0,2] = translation_xy_I2[0]
transform_T[1,2] = translation_xy_I2[1]
transform_T = transform_T[:2, :]

H2 = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ H2

stereo_L = cv2.warpPerspective(I1, H1, (int(W_I1),int(H_I1)))
stereo_R = cv2.warpPerspective(I2, H2, (int(W_I2),int(H_I2)))

plt.imshow(stereo_L)
plt.show()
plt.imshow(stereo_R)
plt.show()