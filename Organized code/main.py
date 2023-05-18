import cv2
import matplotlib.pyplot as plt
import numpy as np
from conf import cfg, load_cfg_fom_args
import os
import open3d as o3d
from image_rectification import rectify
from disparity_map import get_disparity_map , get_depth_map
from middlebury_dataloader import middlebury_DataLoader
from stereo import Two_Image_Stereo
from Utils.ImageUtils import *
from Utils.Triangulation import *
from Utils.FundamentalEssentialMatrix import *
from Utils.MiscUtils import *
from Utils.GeometryUtils import *
from Utils.MathUtils import *
from tqdm import *

def display_depth_map(depth_map, color_img, K1 , K2 , baseline , doffs , ):
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

def depth_2d(image0, image1, K1, K2, baseline, doffs, data_dir):
    f = K1[0 , 0]
    sift = cv2.SIFT_create()
    image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    print("Finding matches")
    kp1, des1 = sift.detectAndCompute(image0_gray, None)
    kp2, des2 = sift.detectAndCompute(image1_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:100]

    matched_pairs = siftFeatures2Array(chosen_matches, kp1, kp2)
    print("Estimating F and E matrix")
    F_best, matched_pairs_inliers = getInliers(matched_pairs)

    E = getEssentialMatrix(K1, K2, F_best)
    R2, C2 = ExtractCameraPose(E)
    pts3D_4 = get3DPoints(K1, K2, matched_pairs_inliers, R2, C2)

    z_count1 = []
    z_count2 = []

    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    for i in range(len(pts3D_4)):
        pts3D = pts3D_4[i]
        pts3D = pts3D/pts3D[3, :]
        x = pts3D[0,:]
        y = pts3D[1, :]
        z = pts3D[2, :]    

        z_count2.append(getPositiveZCount(pts3D, R2[i], C2[i]))
        z_count1.append(getPositiveZCount(pts3D, R1, C1))

    z_count1 = np.array(z_count1)
    z_count2 = np.array(z_count2)
    count_thresh = int(pts3D_4[0].shape[1] / 2)
    idx = np.intersect1d(np.where(z_count1 > count_thresh), np.where(z_count2 > count_thresh))
    R2_ = R2[idx[0]]
    C2_ = C2[idx[0]]
    X_ = pts3D_4[idx[0]]
    X_ = X_/X_[3,:]
    print("Estimated R and C as ", R2_, C2_)
    set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    h1, w1 = image0.shape[:2]
    h2, w2 = image1.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), F_best, imgSize=(w1, h1))
    print("Estimated H1 and H2 as", H1, H2)

    img1_rectified = cv2.warpPerspective(image0, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(image1, H2, (w2, h2))

    set1_rectified = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
    set2_rectified = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)

    img1_rectified_draw = img1_rectified.copy()
    img2_rectified_draw = img2_rectified.copy()

    H2_T_inv =  np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F_best, H1_inv))

    lines1_rectified, lines2_recrified = getEpipolarLines(set1_rectified, set2_rectified, F_rectified, img1_rectified, img2_rectified, "run_output_2d/RectifiedEpilines_" + str(data_dir)+ ".png", True)

    img1_rectified_reshaped = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
    img2_rectified_reshaped = cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))

    img1_rectified_reshaped = cv2.cvtColor(img1_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    img2_rectified_reshaped = cv2.cvtColor(img2_rectified_reshaped, cv2.COLOR_BGR2GRAY)

    window = 6

    left_array, right_array = img1_rectified_reshaped, img2_rectified_reshaped
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        raise "Left-Right image shape mismatch!"
    h, w = left_array.shape
    disparity_map = np.zeros((h, w))

    x_new = w - (2 * window)
    for y in tqdm(range(window, h-window)):
        block_left_array = []
        block_right_array = []
        for x in range(window, w-window):
            block_left = left_array[y:y + window,
                                    x:x + window]
            block_left_array.append(block_left.flatten())

            block_right = right_array[y:y + window,
                                    x:x + window]
            block_right_array.append(block_right.flatten())

        block_left_array = np.array(block_left_array)
        block_left_array = np.repeat(block_left_array[:, :, np.newaxis], x_new, axis=2)

        block_right_array = np.array(block_right_array)
        block_right_array = np.repeat(block_right_array[:, :, np.newaxis], x_new, axis=2)
        block_right_array = block_right_array.T

        abs_diff = np.abs(block_left_array - block_right_array)
        sum_abs_diff = np.sum(abs_diff, axis = 1)
        idx = np.argmin(sum_abs_diff, axis = 0)
        disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity 




    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))
    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    # save 
    plt.savefig('run_output_2d/disparity_image_heat.png')
    # plt.savefig('run_output_2d/disparity_image_heat' +str(data_dir)+ ".png")
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('run_output_2d/disparity_image_gray.png')
    # plt.savefig('run_output_2d/disparity_image_gray' +str(data_dir)+ ".png")

    depth = (baseline * f) / (disparity_map + 1e-10)
    depth[depth > 100000] = 100000

    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('run_output_2d/depth_image_heat.png')
    # plt.savefig('run_output_2d/depth_image' +str(data_dir)+ ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('run_output_2d/depth_image_gray.png')
    # plt.savefig('run_output_2d/depth_image_gray' +str(data_dir)+ ".png")
    plt.show()  
    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)
    Z = depth_map
    ax.scatter(X, Y, Z, c='r', marker='o')
    plt.savefig('run_output_2d/depth_image_3d.png')
    # plt.savefig('run_output_2d/depth_image_3d' +str(data_dir)+ ".png")
    plt.show()


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
    

    if cfg.TASK.TWO_IMAGES_STEREO == True:
        print("Task 1: Two Images Stereo")
        Image_list = middlebury_DataLoader.load_middlebury_data2d(data_dir)
        for obj in Image_list:
            left_img = obj['left_image']
            right_img = obj['right_image']
            K_left = obj['K_left']
            K_right = obj['K_right']
            baseline = obj['baseline']
            doffs = obj['doffs']
            depth_2d(left_img, right_img, K_left, K_right, baseline, doffs , data_dir)
            # stereo_L , stereo_R , _ , _ = rectify(left_img , right_img)
            # disparity_map = get_disparity_map(cfg, stereo_L , stereo_R)
            # depth_map = get_depth_map(disparity_map , K_left , K_right , baseline , doffs)
            # display_depth_map(depth_map , left_img , K_left , K_right , baseline , doffs)
            break
    
    elif cfg.TASK.MULTIPLE_IMAGES_STEREO == True:
        print("Task 2: Multiple Images Stereo")

        Image_list = middlebury_DataLoader.load_middlebury_data3d(data_dir , "dino")
        point_cloud_list = []
        poiint_color_list = []
        disparity_list = []
        depth_list = []

        # create pairs of images for stereo
        image_pairs = []
        i = 0
        # while i < len(Image_list):
        #     image_pairs.append([i , i+2])
        #     i += 3

        image_pairs.append([3 , 1])
        # image_pairs.append([5 , 2])
        # image_pairs.append([6 , 4])

        for pair in image_pairs:
            print("Calculating maps and point cloud for pair:", pair)
            i = pair[0]
            j = pair[1]
            img1 = Image_list[i]
            img2 = Image_list[j]
            point_cloud , point_color , disparity_map , depth_map =  Two_Image_Stereo(img1, img2, 5)
            point_cloud_list.append(point_cloud)
            poiint_color_list.append(point_color)
            disparity_list.append(disparity_map)
            depth_list.append(depth_map)
        
        pcl = np.concatenate(point_cloud_list, axis=0)
        pcl_color = np.concatenate(poiint_color_list, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(pcl_color / 255.0)
        o3d.visualization.draw_geometries([pcd])
        # save the point cloud for visualization
        o3d.io.write_point_cloud("run_output_3d/point_cloud_rt1.ply", pcd)
        
if __name__ == "__main__":
    main()