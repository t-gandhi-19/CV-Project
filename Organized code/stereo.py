import numpy as np
import cv2
from kernel import sum_abs_dist
import open3d as o3d

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

def Depth_Map(Disparity_map , K_corrected_left,  Baseline):
    depth_map = np.zeros_like(Disparity_map, dtype=np.float32)
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            if Disparity_map[y, x] > 0:
                depth_map[y, x] = (K_corrected_left[0, 0] * Baseline) / Disparity_map[y, x]
    u, v = np.meshgrid(np.arange(Disparity_map.shape[1]), np.arange(Disparity_map.shape[0]))
    xyz_cam = np.dstack(((u - K_corrected_left[0, 2]) * depth_map / K_corrected_left[0, 0], 
                         (v - K_corrected_left[1, 2]) * depth_map / K_corrected_left[1, 1], 
                         depth_map))
    return depth_map, xyz_cam

    # dep_map = np.divide( K_corrected_left[1, 1] * Baseline, Disparity_map )
    # u, v = np.meshgrid( np.arange(Disparity_map.shape[1]), np.arange(Disparity_map.shape[0]) )
    # xyz_cam = np.dstack( ( (u - K_corrected_left[0, 2]) * dep_map / K_corrected_left[0, 0], (v - K_corrected_left[1, 2]) * dep_map / K_corrected_left[1, 1], dep_map ) )
    # return dep_map, xyz_cam

def image_patch(image , patch_size):
    image = image.astype(np.float32)/255.0

    padded_image = np.empty( ( image.shape[0] + patch_size - 1, image.shape[1] + patch_size - 1, image.shape[2] ) )
    for i in range(3):
        padded_image[:, :, i] = np.pad( image[:, :, i], int( patch_size / 2 ), mode='constant' )
    patch = np.zeros( (image.shape[0], image.shape[1], patch_size*patch_size, 3) )
    for x in range( image.shape[1] ):
        for y in range( image.shape[0] ):
            index_y, index_x = np.meshgrid( np.arange( x - int( patch_size / 2 ), x + int( patch_size / 2 ) + 1 ), np.arange( y - int( patch_size / 2 ), y + int( patch_size / 2 ) + 1 ) )
            index_x += int( patch_size / 2 )
            index_y += int( patch_size / 2 )
            for i in range(3):
                patch[y, x, :, i] = padded_image[ index_x, index_y, i ].flatten()
    return patch

def Disparity_Map(Rectified_left_image , Rectified_right_image , Kernel_size=5 , d0 = 5 , EPS = 1e-3):
    print("Computing Disparity Map")
    h = Rectified_left_image.shape[0]
    w = Rectified_left_image.shape[1]
    disparity_map = np.zeros((h , w)).astype(np.float32)
    consistency_map = np.zeros((h , w)).astype(np.float32)

    patch_size = Kernel_size
    patch_left = image_patch(Rectified_left_image , patch_size)
    patch_right = image_patch(Rectified_right_image , patch_size) 

    v_left_index , v_right_index = np.arrange(h) , np.arrange(h)
    disparity_points = v_left_index[:, None] - v_right_index[None, :] + d0

    iter = 0
    while iter < w:
        buff_left = patch_left[:, iter] 
        buff_right = patch_right[:, iter]
        value = sum_abs_dist(buff_left , buff_right)
        best_match_right = np.argmin(value , axis=1)
        match = np.arrange(h)
        disparity_map[:, iter] = disparity_points[match , best_match_right]
        best_match_left = np.argmin(value[: , best_match_right] , axis=0)
        if best_match_left == v_left_index:
            consistency_value = 1
        else:
            consistency_value = 0
        consistency_map[:, iter] = consistency_value
        iter += 1
    
    return disparity_map , consistency_map

def Point_Cloud_refiner(Depth_Map , Rectified_left_image , Points_cloud , Rectification_R_left , Rectification_T_right , Consitency_Map , z_near=0.1 , z_far=1000):
    print("Refining Point Cloud")
    # remove background 
    hsv_mask = cv2.cvtColor(Rectified_left_image , cv2.COLOR_RGB2HSV)[..., -1]
    hsv_mask = (hsv_mask > 51).astype(np.uint8) * 255
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel).astype(float)
    mask_depth = ((Depth_Map > z_near) & (Depth_Map < z_far)).astype(float)
    mask = np.minimum(hsv_mask, mask_depth)
    if Consitency_Map is not None:
        mask = np.minimum(mask, Consitency_Map)
    
    camera_points = Points_cloud.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(camera_points.reshape(-1, 3).copy())
    cl , ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    _pcl_mask = np.zeros(camera_points.shape[0])
    _pcl_mask[ind] = 1
    point_cloud_mask = np.zeros(Points_cloud.shape[0] * Points_cloud.shape[1])
    point_cloud_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = point_cloud_mask.reshape(Points_cloud.shape[0], Points_cloud.shape[1])
    mask = np.minimum(mask, mask_pcl)

    camera_points = Points_cloud.reshape(-1, 3)[mask.reshape(-1) > 0]
    camera_colors = Rectified_left_image.reshape(-1, 3)[mask.reshape(-1) > 0]
    world_points = (Rectification_R_left.T @ (camera_points.T - Rectification_T_right)).T

    return mask , world_points , camera_points , camera_colors

def Rectificator(left_image , right_image , Left_Rectified_R , Right_Rectified_R , Left_K , Right_K , x_axis_padding= 20 , y_axis_padding= 20):
    # Given the rectify rotation, compute the rectified view and corrected projection matrix
    if left_image.shape == right_image.shape :
        h, w = left_image.shape[:2]

        Homography_matrix_left = Left_K @ Left_Rectified_R @ np.linalg.inv(Left_K)
        Homography_matrix_right = Right_K @ Right_Rectified_R @ np.linalg.inv(Right_K)
        left_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        right_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        left_rectified_corners = cv2.perspectiveTransform(left_corners, Homography_matrix_left).squeeze(1)
        right_rectified_corners = cv2.perspectiveTransform(right_corners, Homography_matrix_right).squeeze(1)
        u_left_min , v_left_min = np.int32(left_rectified_corners.min(axis=0))
        u_left_max , v_left_max = np.int32(left_rectified_corners.max(axis=0))
        u_right_min , v_right_min = np.int32(right_rectified_corners.min(axis=0))
        u_right_max , v_right_max = np.int32(right_rectified_corners.max(axis=0))

        w_max = int( np.floor( max(u_left_max, u_right_max) ) ) - x_axis_padding * 2
        h_max = int( np.floor( min(v_left_max - v_left_min, v_right_max - v_right_min) ) ) - y_axis_padding * 2

        assert Left_K[0, 2] == Right_K[0, 2]
        K_i_corr, K_j_corr = Left_K.copy(), Right_K.copy()
        K_i_corr[0, 2] -= x_axis_padding
        K_i_corr[1, 2] -= v_left_min + y_axis_padding
        K_j_corr[0, 2] -= x_axis_padding
        K_j_corr[1, 2] -= v_right_min + y_axis_padding
        
        Left_rectified_Image = cv2.warpPerspective(left_image, ( K_i_corr @ Left_Rectified_R @ ( np.linalg.inv( Left_K ) ) ), dsize=( w_max, h_max ) )
        Right_rectified_Image = cv2.warpPerspective(right_image, ( K_j_corr @ Right_Rectified_R @ ( np.linalg.inv( Right_K ) ) ), dsize=( w_max, h_max ) )
    
        return Left_rectified_Image, Right_rectified_Image, K_i_corr, K_j_corr
    else:
        print("images are not the same size")
        raise ValueError

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

    Rectification_R_left = find_Rectification_R(EPS , Relative_T)
    Rectification_R_right = Rectification_R_left @ Relative_R
    Rectification_T_right = Rectification_R_left @ Relative_T
    Rectified_left_image , Rectified_right_image , K_corrected_left , K_corrected_right = Rectificator(left["Image"] , right["Image"] , Rectification_R_left , Rectification_R_right , left["K"] , right["K"])

    if Rectified_left_image.shape != Rectified_right_image.shape:
        print("images are not the same size")
        raise ValueError

    if K_corrected_left.shape != K_corrected_right.shape:
        print("K matrices are not the same size")
        raise ValueError
    
    d0 = K_corrected_right[1,2] - K_corrected_left[1,2]
    Disparity_map , Consitency_Map = Disparity_Map(Rectified_left_image , Rectified_right_image , Kernel_size=Kernel_size , d0 = d0)
    Depth_map , Points_cloud = Depth_Map(Disparity_map , K_corrected_left,  Baseline)
    mask , World_point_cloud , Camera_point_cloud , Point_cloud_colors = Point_Cloud_refiner(Depth_Map , Rectified_left_image , Points_cloud , Rectification_R_left , Rectification_T_right , Consitency_Map , z_near=0.1 , z_far=1000)
    return World_point_cloud , Point_cloud_colors , Disparity_map , Depth_map