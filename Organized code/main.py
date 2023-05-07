import cv2
import matplotlib.pyplot as plt
import numpy as np
from conf import cfg, load_cfg_fom_args
import os
from image_rectification import rectify
from disparity_map import get_disparity_map

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
    # print("List of folders in data directory:", list_of_folders)

    if cfg.DATASET.RECTIFIED == False:
        for i, name in enumerate(list_of_folders):
            l_img = cv2.imread(os.path.join(data_dir, name, "left.jpg"))
            r_img = cv2.imread(os.path.join(data_dir, name, "right.jpg"))
            rectified_l_img , rectified_r_img  , _ , _ = rectify(l_img, r_img) 
            cv2.imwrite(os.path.join(data_dir, name, "rectified_left.jpg"), rectified_l_img)
            cv2.imwrite(os.path.join(data_dir, name, "rectified_right.jpg"), rectified_r_img)
    
    disparity_map = get_disparity_map(cfg)

if __name__ == "__main__":
    main()