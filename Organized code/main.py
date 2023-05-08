import cv2
import matplotlib.pyplot as plt
import numpy as np
from conf import cfg, load_cfg_fom_args
import os
import open3d as o3d
from image_rectification import rectify
from disparity_map import get_disparity_map , get_depth_map

def display_depth_map(depth_map, color_img, K1 , K2 , baseline , doffs):
    '''
	Parameters:
		fx-- focal length in x dir (scaled if resized)
		fy-- focal length in y dir (scaled if resized)
		cx-- x axis principle point (scaled if resized)
		cy-- y axis principle point (scaled if resized)
	Displays an Open3D point cloud
	'''
    fx = K1[0]
    fy = K1[4]
    cx = K1[2]
    cy = K1[5]
    
    shape = color_img.shape;
    h = shape[0]
    w = shape[1]
	
    img = o3d.geometry.Image(color_img.astype('uint8'))
    depth = o3d.geometry.Image(depth_map.astype('uint16'))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
    o3d_pinhole = o3d.camera.PinholeCameraIntrinsic()
    o3d_pinhole.set_intrinsics(w, h, fx, fy, cx, cy)

    pcd_from_depth_map = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pinhole)
    pcd_from_depth_map.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_from_depth_map])

def get_parameters(cfg , path_to_calibration_file):
    calibration_file = open(path_to_calibration_file, "r")
    calibration_file = calibration_file.readlines()
    calibration_file = [x.strip() for x in calibration_file]
    # first line is intriisc parameters
    # K1 is between [ and ]
    K1 = calibration_file[0].split("[")[1].split("]")[0]
    K2 = calibration_file[1].split("[")[1].split("]")[0]
    doffs = float(calibration_file[2].split("=")[1])
    baseline = float(calibration_file[3].split("=")[1])
    K1 = K1.split(" ")
    K1 = [x.split(";") for x in K1]
    K1 = [item for sublist in K1 for item in sublist]
    K1 = [x for x in K1 if x]
    K1 = np.array(K1)
    K2 = K2.split(" ")
    K2 = [x.split(";") for x in K2]
    K2 = [item for sublist in K2 for item in sublist]
    K2 = [x for x in K2 if x]
    K2 = np.array(K2)
    K1 = K1.astype(np.float64)
    K2 = K2.astype(np.float64)
    return K1 , K2 , doffs , baseline


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
   
    # remove .DS_Store from list of folders
    if ".DS_Store" in list_of_folders:
        list_of_folders.remove(".DS_Store")

    list_of_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print("List of folders in data directory:", list_of_folders)

    for i, name in enumerate(list_of_folders):
        if cfg.DATASET.RECTIFIED == False:
            l_img = cv2.imread(os.path.join(data_dir, name, "im0.png"))
            r_img = cv2.imread(os.path.join(data_dir, name, "im1.png"))
            rectified_l_img , rectified_r_img  , _ , _ = rectify(l_img, r_img) 
            cv2.imwrite(os.path.join(data_dir, name, "rectified_im0.png"), rectified_l_img)
            cv2.imwrite(os.path.join(data_dir, name, "rectified_im1.png"), rectified_r_img)
        else:
            rectified_l_img = cv2.imread(os.path.join(data_dir, name, "im0.png"))
            rectified_r_img = cv2.imread(os.path.join(data_dir, name, "im1.png"))
        
        calibration_file = os.path.join(data_dir, name, "calib.txt")
        K1 , K2 , doffs , baseline = get_parameters(cfg , calibration_file)
        disparity_map = get_disparity_map(cfg , rectified_l_img , rectified_r_img)
        depth_map = get_depth_map(disparity_map , K1 , K2 , doffs , baseline)

if __name__ == "__main__":
    main()