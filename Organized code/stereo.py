import numpy as np
import cv2


import cv2
import numpy as np

def Find_Relative_R_T_B_using_F(Im1 , Im2 , K1 , K2):
    # find corresponding points in both images using SIFT and RANSAC to find F then find R,T,B
    sift = cv2.SIFT_create()
    kp1 , des1 = sift.detectAndCompute(Im1 , None)
    kp2 , des2 = sift.detectAndCompute(Im2 , None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE , trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params , search_params)
    matches = flann.knnMatch(des1 , des2 , k=2)
    good_matches = []
    pts1 = []
    pts2 = []
    for i , (u , v) in enumerate(matches):
        if u.distance < 0.75 * v.distance:
            good_matches.append(u)
            pts2.append(kp2[u.trainIdx].pt)
            pts1.append(kp1[u.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F , mask = cv2.findFundamentalMat(pts1 , pts2 , cv2.FM_RANSAC)
    E = K2.T @ F @ K1
    _ , R , T , _ = cv2.recoverPose(E , pts1 , pts2 , K1)
    T = T.flatten()
    B = np.linalg.norm(T)
    # T = T / B
    return R , T , B

def Find_Relative_R_T_B(Im1_R , Im1_T , Im2_R , Im2_T):
    R_ji = Im1_R @ np.linalg.inv( Im2_R )
    T_ji = -R_ji @ Im2_T + Im1_T
    B = np.linalg.norm( T_ji )
    return R_ji, T_ji, B

def find_Rectification_R(EPS= None , Relative_T = None):
    pass

def Two_Image_Stereo(Image1 , Image2 , Kernel_size=5 , EPS= 1e-6):
    Im1_R = Image1["R"]
    Im1_T = Image1["T"]

    Im2_R = Image2["R"]
    Im2_T = Image2["T"]

    Relative_R = None
    Relative_T = None
    Baseline = None

    Relative_R , Relative_T , Baseline = Find_Relative_R_T_B(Im1_R , Im1_T , Im2_R , Im2_T)
    left = None
    right = None
    if Relative_T[2] < 0: 
        print("images are not left and right respectively")
        left = Image2
        right = Image1
        Relative_R , Relative_T , Baseline = Find_Relative_R_T_B(Im2_R , Im2_T , Im1_R , Im1_T)
    else:
        left = Image1
        right = Image2

    Rectification_R = find_Rectification_R(EPS , Relative_T)
    

    # Relative_R2 , Relative_T2 , Baseline2 = Find_Relative_R_T_B_using_F(Image1["Image"] , Image2["Image"] , Image1["K"] , Image2["K"])
    # Relative_R3 , Relative_T3 , Baseline3 = Find_Relative_R_T_B3(Image1["Image"] , Image2["Image"] , Image1["K"] , Image2["K"])
    
    # print("Relative_R = " , Relative_R)
    # print("Relative_R2 = " , Relative_R2)
    # print("Relative_R3 = " , Relative_R3)

    # print("\n")

    # print("Relative_T = " , Relative_T)
    # print("Relative_T2 = " , Relative_T2)
    # print("Relative_T3 = " , Relative_T3)

    # print("\n")

    # print("Baseline = " , Baseline)
    # print("Baseline2 = " , Baseline2)
    # print("Baseline3 = " , Baseline3)