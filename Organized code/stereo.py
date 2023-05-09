import numpy as np
import cv2

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
    _ , R , T , _ = cv2.recoverPose(E , pts1 , pts2 , K2)
    B = np.linalg.norm(T)
    T = T / B
    return R , T , B


def Find_Relative_R_T_B(Im1_R , Im1_T , Im2_R , Im2_T):
    R_ji = Im1_R @ np.linalg.inv( Im2_R )
    T_ji = -R_ji @ Im2_T + Im1_T
    B = np.linalg.norm( T_ji )
    return R_ji, T_ji, B


def Two_Image_Stereo(Image1 , Image2 , Kernel_size=5 , Kernel_Method=None):
    Im1_R = Image1["R"]
    Im1_T = Image1["T"]

    Im2_R = Image2["R"]
    Im2_T = Image2["T"]

    Relative_R , Relative_T , Baseline = Find_Relative_R_T_B(Im1_R , Im1_T , Im2_R , Im2_T)
    print("Relative_R = " , Relative_R)
    print("Relative_T = " , Relative_T)
    print("Baseline = " , Baseline)

    Relative_R2 , Relative_T2 , Baseline2 = Find_Relative_R_T_B_using_F(Image1["Image"] , Image2["Image"] , Image1["K"] , Image2["K"])
    print("Relative_R2 = " , Relative_R2)
    print("Relative_T2 = " , Relative_T2)
    print("Baseline2 = " , Baseline2)
    

def compute_rectification_R(EPS, T_ji):
    '''T_ji - 3x1'''
    
    e_i = T_ji.squeeze(-1) / ( T_ji.squeeze(-1)[1] + EPS )
    # squeezed_T_ji = []
    # for i in range(len(T_ji)):
    #     if T_ji.shape[i] != 1:
    #         squeezed_T_ji.append(T_ji[i])
    
    # divisor = squeezed_T_ji[1] + EPS     
    # e_i = []
    # for i in range(len(squeezed_T_ji)):
    #     e_i.append(squeezed_T_ji[i] / divisor)
    
    es = []   
    norm = [np.sqrt(np.sum(np.square(T_ji + EPS)))]
    e_2 = T_ji / norm[0]
    es.append(e_2)
    
    e_1 = np.zeros((3,1))
    e_1[0] = e_2[1]*1
    e_1[1] = -e_2[0]*1
    e_1[2] = 0
    norm.append(np.sqrt(np.sum(np.square(e_1 + EPS))))
    e_1 = e_1 / norm[1]
    es.append(e_1)
    
    e_3 = np.zeros(3)
    e_3[0] = e_1[1]*e_2[2] - e_1[2]*e_2[1]
    e_3[1] = e_1[2]*e_2[0] - e_1[0]*e_2[2]
    e_3[2] = e_1[0]*e_2[1] - e_1[1]*e_2[0]
    norm.append(np.sqrt(np.sum(np.square(e_3 + EPS))))
    e_3 = e_3 / norm[2]
    es.append(e_3)
    
    R_irect = np.zeros((3, 3))
    print(es[1].shape)
    R_irect[0, :] = es[1].reshape((3))
    R_irect[1, :] = es[0].reshape((3))
    R_irect[2, :] = es[2].reshape((3))
    
    return R_irect
    
