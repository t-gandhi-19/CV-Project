import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.notebook import tqdm
from conf import cfg, load_cfg_fom_args
import os

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
    inlier_kp1 = []
    inlier_kp2 = []
    inlier_matches = []
    for i in range(len(inlier_mask))
        if i[0] == 0:
            inlier_kp1.extend(best_kp1[i])
            inlier_kp2.extend(best_kp2[i])
            inlier_matches.extend(best_matches[i])

def main():
    print("Hello World!")
    load_cfg_fom_args(description="Victim_training")
    output_dir = cfg.SAVE_DIR
    output_dir_exists = os.path.exists(output_dir)
    if output_dir_exists:
        print("Output directory already exists!" , output_dir)
    else:
        os.makedirs(output_dir)
        print("Output directory created!", output_dir)

    # list of all folders in data directory
    data_dir = cfg.DATASET.DATA_DIR
    data_dir_exists = os.path.exists(data_dir)
    list_of_folders = []
    if data_dir_exists:
        print("Data directory exists!", data_dir)
        list_of_folders.extend(os.listdir(data_dir))

    else:
        print("Data directory does not exist!", data_dir)
   
    list_of_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print("List of folders in data directory:", list_of_folders)

    for i, name in enumerate(list_of_folders):
        l_img = cv2.imread(os.path.join(data_dir, name, "left.jpg"))
        r_img = cv2.imread(os.path.join(data_dir, name, "right.jpg"))
        rectified_l_img , rectified_r_img = rectify(l_img, r_img) 
        cv2.imwrite(os.path.join(data_dir, name, "rectified_left.jpg"), rectified_l_img)
        cv2.imwrite(os.path.join(data_dir, name, "rectified_right.jpg"), rectified_r_img)

if __name__ == "__main__":
    main()