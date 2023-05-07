import cv2
import numpy as np
import random

def rectify(l_img, r_img):
    left_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

    # Find the stereo correspondence...
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_gray,None)
    kp2, des2 = orb.detectAndCompute(right_gray,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    best_kp1 = []
    best_kp2 = []
    best_matches = []

    for m in matches:
        best_kp1.append(kp1[m.queryIdx].pt)
        best_kp2.append(kp2[m.trainIdx].pt)
        best_matches.append(m)
    
    best_kp1 = np.array(best_kp1)
    best_kp2 = np.array(best_kp2)
    best_matches = np.array(best_matches)

    F, inlier_mask = cv2.findFundamentalMat(best_kp1, best_kp2, cv2.FM_7POINT)
    # inlier_mask = inlier_mask.flatten()
    inlier_kp1 = []
    inlier_kp2 = []
    for i in range(len(inlier_mask)):
        if inlier_mask[i][0] == 1:
            inlier_kp1.extend(best_kp1[i])
            inlier_kp2.extend(best_kp2[i])


    # stereo rectifaction uncalibrated
    _ , H1, H2 = cv2.stereoRectifyUncalibrated(np.float64(inlier_kp1), np.float64(inlier_kp2), F,left_gray.shape[::-1])
    I1_rectified = np.float64([[[0,0], [left_gray.shape[1],0], [left_gray.shape[1],left_gray.shape[0]], [0,left_gray.shape[0]]]])
    warped_I1_rectified = cv2.perspectiveTransform(I1_rectified, H1)

    I2_rectified = np.float64([[[0,0], [right_gray.shape[1],0], [right_gray.shape[1],right_gray.shape[0]], [0,right_gray.shape[0]]]])
    warped_I2_rectified = cv2.perspectiveTransform(I2_rectified, H2)

    min_x = {}
    min_y = {}
    max_x = {}
    max_y = {}
   
    min_x['I1'] = min(warped_I1_rectified[0][0][0], warped_I1_rectified[0][1][0], warped_I1_rectified[0][2][0], warped_I1_rectified[0][3][0])
    min_y['I1'] = min(warped_I1_rectified[0][0][1], warped_I1_rectified[0][1][1], warped_I1_rectified[0][2][1], warped_I1_rectified[0][3][1])
    max_x['I1'] = max(warped_I1_rectified[0][0][0], warped_I1_rectified[0][1][0], warped_I1_rectified[0][2][0], warped_I1_rectified[0][3][0])
    max_y['I1'] = max(warped_I1_rectified[0][0][1], warped_I1_rectified[0][1][1], warped_I1_rectified[0][2][1], warped_I1_rectified[0][3][1])

    min_x['I2'] = min(warped_I2_rectified[0][0][0], warped_I2_rectified[0][1][0], warped_I2_rectified[0][2][0], warped_I2_rectified[0][3][0])
    min_y['I2'] = min(warped_I2_rectified[0][0][1], warped_I2_rectified[0][1][1], warped_I2_rectified[0][2][1], warped_I2_rectified[0][3][1])
    max_x['I2'] = max(warped_I2_rectified[0][0][0], warped_I2_rectified[0][1][0], warped_I2_rectified[0][2][0], warped_I2_rectified[0][3][0])
    max_y['I2'] = max(warped_I2_rectified[0][0][1], warped_I2_rectified[0][1][1], warped_I2_rectified[0][2][1], warped_I2_rectified[0][3][1])

    I1_translation = np.array([max(0 , -min_x['I1']), max(0 , -min_y['I1'])])
    I2_translation = np.array([max(0 , -min_x['I2']), max(0 , -min_y['I2'])])

    I1_width = int(max_x['I1'] + I1_translation[0])
    I1_height = int(max_y['I1'] + I1_translation[1])

    I2_width = int(max_x['I2'] + I2_translation[0])
    I2_height = int(max_y['I2'] + I2_translation[1])

    transform_I1 = np.array([[1, 0, I1_translation[0]], [0, 1, I1_translation[1]], [0, 0, 1]])
    transform_I2 = np.array([[1, 0, I2_translation[0]], [0, 1, I2_translation[1]], [0, 0, 1]])
    
    H1 = np.matmul(transform_I1, H1)
    H2 = np.matmul(transform_I2, H2)

    stereo_L = cv2.warpPerspective(l_img, H1, (I1_width, I1_height))
    stereo_R = cv2.warpPerspective(r_img, H2, (I2_width, I2_height))

    return stereo_L, stereo_R, H1, H2
