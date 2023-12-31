import cv2
import numpy as np
import open3d as o3d
import src.sfm2view as sfm
from environment import ApplicationProperties

class TwoViewStereo( object ):

    @staticmethod
    def homo_corners( h, w, H ):
        corners_bef = np.float32( [[0, 0], [w, 0], [w, h], [0, h]] ).reshape( -1, 1, 2 )
        corners_aft = cv2.perspectiveTransform( corners_bef, H ).squeeze(1)
        u_min, v_min = corners_aft.min( axis=0 )
        u_max, v_max = corners_aft.max( axis=0 )
        return u_min, u_max, v_min, v_max

    @staticmethod
    def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
        """Given the rectify rotation, compute the rectified view and corrected projection matrix

        Parameters
        ----------
        rgb_i,rgb_j : [H,W,3]
        R_irect,R_jrect : [3,3]
            p_rect_left = R_irect @ p_i
            p_rect_right = R_jrect @ p_j
        K_i,K_j : [3,3]
            original camera matrix
        u_padding,v_padding : int, optional
            padding the border to remove the blank space, by default 20

        Returns
        -------
        [H,W,3],[H,W,3],[3,3],[3,3]
            the rectified images
            the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
        """
        # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
        assert rgb_i.shape == rgb_j.shape
        
        h, w = rgb_i.shape[:2]

        ui_min, ui_max, vi_min, vi_max = TwoViewStereo.homo_corners( h, w, K_i @ R_irect @ np.linalg.inv(K_i) )
        uj_min, uj_max, vj_min, vj_max = TwoViewStereo.homo_corners( h, w, K_j @ R_jrect @ np.linalg.inv(K_j) )

        # The distortion on u direction (the world vertical direction) is minor, ignore this
        w_max = int( np.floor( max(ui_max, uj_max) ) ) - u_padding * 2
        h_max = int( np.floor( min(vi_max - vi_min, vj_max - vj_min) ) ) - v_padding * 2

        assert K_i[0, 2] == K_j[0, 2]
        K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
        K_i_corr[0, 2] -= u_padding
        K_i_corr[1, 2] -= vi_min + v_padding
        K_j_corr[0, 2] -= u_padding
        K_j_corr[1, 2] -= vj_min + v_padding

        rgb_i_rect = cv2.warpPerspective(rgb_i, ( K_i_corr @ R_irect @ ( np.linalg.inv( K_i ) ) ), dsize=( w_max, h_max ) )
        rgb_j_rect = cv2.warpPerspective(rgb_j, ( K_j_corr @ R_jrect @ ( np.linalg.inv( K_j ) ) ), dsize=( w_max, h_max ) )

        return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr

    @staticmethod
    def compute_right2left_transformation( R_wi, T_wi, R_wj, T_wj ):
        """Compute the transformation that transform the coordinate from j coordinate to i

        Parameters
        ----------
        R_wi, R_wj : [3,3]
        T_wi, T_wj : [3,1]
            p_i = R_wi @ p_w + T_wi
            p_j = R_wj @ p_w + T_wj
        Returns
        -------
        [3,3], [3,1], float
            p_i = R_ji @ p_j + T_ji, B is the baseline
        """

        R_ji = R_wi @ np.linalg.inv( R_wj )
        T_ji = -R_ji @ T_wj + T_wi
        B = np.linalg.norm( T_ji )

        return R_ji, T_ji, B

    @staticmethod
    def compute_rectification_R( EPS, T_ji ):
        """Compute the rectification Rotation

        Parameters
        ----------
        T_ji : [3,1]

        Returns
        -------
        [3,3]
            p_rect = R_irect @ p_i
        """
        # check the direction of epipole, should point to the positive direction of y axis
        e_i = T_ji.squeeze(-1) / ( T_ji.squeeze(-1)[1] + EPS )
        
        e_2 = ( T_ji / np.linalg.norm( T_ji + EPS ) ).flatten()
        
        e_1 = np.cross( e_2, np.array([0, 0, 1]) )
        e_1 = e_1 / np.linalg.norm( e_1 + EPS )

        e_3 = np.cross( e_1, e_2 )
        e_3 = e_3 /  np.linalg.norm( e_3 + EPS )

        R_irect = np.vstack((e_1, e_2, e_3))
        
        return R_irect

    @staticmethod
    def ssd_kernel(src, dst):
        """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

        Parameters
        ----------
        src : [M,K*K,3]
            M left view patches
        dst : [N,K*K,3]
            N right view patches

        Returns
        -------
        [M,N]
            error score for each left patches with all right patches.
        """

        assert src.ndim == 3 and dst.ndim == 3
        assert src.shape[1:] == dst.shape[1:]

        # Creating M * N * (K*K) * 3 matrix for each of these patches
        src = src[:, np.newaxis, :, :]
        dst = dst[np.newaxis, :, :, :]
        ssd = np.zeros( ( src.shape[0], dst.shape[1] ) )
        for i in range( 3 ):
            ssd += np.sum( np.square( src[:, :, :, i] - dst[:, :, :, i] ) , axis = 2 )

        return ssd

    @staticmethod
    def sad_kernel(src, dst):
        """Compute SAD Error, the RGB channels should be treated saperately and finally summed up

        Parameters
        ----------
        src : [M,K*K,3]
            M left view patches
        dst : [N,K*K,3]
            N right view patches

        Returns
        -------
        [M,N]
            error score for each left patches with all right patches.
        """

        assert src.ndim == 3 and dst.ndim == 3
        assert src.shape[1:] == dst.shape[1:]

        src = src[:, np.newaxis, :, :]
        dst = dst[np.newaxis, :, :, :]
        sad = np.zeros( ( src.shape[0], dst.shape[1] ) )
            
        for i in range( 3 ):
            sad += np.sum( np.abs( src[:, :, :, i] - dst[:, :, :, i] ) , axis = 2 )

        return sad

    @staticmethod
    def zncc_kernel(src, dst):
        """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

        Parameters
        ----------
        src : [M,K*K,3]
            M left view patches
        dst : [N,K*K,3]
            N right view patches

        Returns
        -------
        [M,N]
            score for each left patches with all right patches.
        """
        EPS = 1e-6
        assert src.ndim == 3 and dst.ndim == 3
        assert src.shape[1:] == dst.shape[1:]

        zncc = np.zeros( ( src.shape[0], dst.shape[0] ) )
        for i in range( 3 ):
            src_mean = np.mean( src[:, :, i], axis = 1 ).reshape( (-1, 1) )
            src_sig = np.std( src[:, :, i], axis = 1 ).reshape( (-1, 1) )
            dst_mean = np.mean( dst[:, :, i], axis = 1 ).reshape( (-1, 1) )
            dst_sig = np.std( dst[:, :, i], axis = 1 ).reshape( (-1, 1) )

            w_1 = ( src[:, :, i] - src_mean )[:, np.newaxis, :]
            w_2 = ( dst[:, :, i] - dst_mean )[np.newaxis, :, :]
            
            zncc += np.sum( w_1 * w_2, axis=2 ) / ( (src_sig @ dst_sig.T) + EPS )

        # ! note here we use minus zncc since we use argmin outside, but the zncc is a similarity, which should be maximized
        return zncc * (-1.0)

    @staticmethod
    def image2patch(image, k_size):
        """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

        Parameters
        ----------
        image : [H,W,3]
        k_size : int, must be odd number; your function should work when k_size = 1

        Returns
        -------
        [H,W,k_size**2,3]
            The patch buffer for each pixel
        """

        padded_image = np.empty( ( image.shape[0] + k_size - 1, image.shape[1] + k_size - 1, image.shape[2] ) )
        for i in range(3):
            padded_image[:, :, i] = np.pad( image[:, :, i], int( k_size / 2 ), mode='constant' )
        patch = np.zeros( (image.shape[0], image.shape[1], k_size*k_size, 3) )

        for x in range( image.shape[1] ):
            for y in range( image.shape[0] ):
                index_y, index_x = np.meshgrid( np.arange( x - int( k_size / 2 ), x + int( k_size / 2 ) + 1 ), np.arange( y - int( k_size / 2 ), y + int( k_size / 2 ) + 1 ) )
                index_x += int( k_size / 2 )
                index_y += int( k_size / 2 )
                for i in range(3):
                    patch[y, x, :, i] = padded_image[ index_x, index_y, i ].flatten()

        return patch

    @staticmethod
    def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
        """Compute the disparity map from two rectified view

        Parameters
        ----------
        rgb_i,rgb_j : [H,W,3]
        d0 : see the hand out, the bias term of the disparty caused by different K matrix
        k_size : int, optional
            The patch size, by default 3
        kernel_func : function, optional
            the kernel used to compute the patch similarity, by default ssd_kernel
        img2patch_func : function, optional
            this is for auto-grader purpose, in grading, we will use our correct implementation of the image2path function to exclude double count for errors in image2patch function

        Returns
        -------
        disp_map: [H,W], dtype=np.float64
            The disparity map, the disparity is defined in the handout as d0 + vL - vR

        lr_consistency_mask: [H,W], dtype=np.float64
            For each pixel, 1.0 if LR consistent, otherwise 0.0
        """

        h, w = rgb_i.shape[:2]
        disp_map = np.empty( ( h, w ), dtype = np.float64 )
        lr_consistency_mask = np.zeros( ( h, w ), dtype = np.float64 )

        patches_i = img2patch_func( rgb_i.astype(float) / 255.0, k_size )
        patches_j = img2patch_func( rgb_j.astype(float) / 255.0, k_size )

        vi_idx, vj_idx = np.arange(h), np.arange(h)
        disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0

        for i in range(w):
            buf_i, buf_j = patches_i[:, i], patches_j[:, i]
            EPS = 1e-6
            value = kernel_func( buf_i, buf_j)
            best_matched_right_pixel = np.argmin( value, axis = 1 )
            match = np.arange( h )
            disp_map[:, i] = disp_candidates[ match, best_matched_right_pixel ]
            best_matched_left_pixel = np.argmin( value[:, best_matched_right_pixel] , axis = 0 )
            consistent_flag = best_matched_left_pixel == vi_idx
            lr_consistency_mask[:, i] = consistent_flag

        return disp_map, lr_consistency_mask

    @staticmethod
    def compute_dep_and_pcl(disp_map, B, K):
        """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
        compute the depth map and backprojected point cloud

        Parameters
        ----------
        disp_map : [H,W]
            disparity map
        B : float
            baseline
        K : [3,3]
            camera matrix

        Returns
        -------
        [H,W]
            dep_map
        [H,W,3]
            each pixel is the xyz coordinate of the back projected point cloud in camera frame
        """

        dep_map = np.divide( K[1, 1] * B, disp_map )

        u, v = np.meshgrid( np.arange(disp_map.shape[1]), np.arange(disp_map.shape[0]) )
        xyz_cam = np.dstack( ( (u - K[0, 2]) * dep_map / K[0, 0], (v - K[1, 2]) * dep_map / K[1, 1], dep_map ) )

        return dep_map, xyz_cam

    @staticmethod
    def postprocess(
        dep_map,
        rgb,
        xyz_cam,
        R_wc,
        T_wc,
        consistency_mask=None,
        hsv_th=45,
        hsv_close_ksize=11,
        z_near=0.45,
        z_far=0.65,
    ):
        """
        Your goal in this function is:
        given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
        compute the pcl_world with shape[N,3] in the world coordinate
        """

        # extract mask from rgb to remove background
        mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
        mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
        # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
        # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

        # constraint z-near, z-far
        mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
        # imageio.imsave("./debug_dep_mask.png", mask_dep)

        mask = np.minimum(mask_dep, mask_hsv)
        if consistency_mask is not None:
            mask = np.minimum(mask, consistency_mask)
        # imageio.imsave("./debug_before_xyz_mask.png", mask)

        # filter xyz point cloud
        pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
        cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        _pcl_mask = np.zeros(pcl_cam.shape[0])


        _pcl_mask[ind] = 1.0
        pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
        pcl_mask[mask.reshape(-1) > 0] = _pcl_mask

        mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
        mask = np.minimum(mask, mask_pcl)
        pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
        pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]
        print("pcl_cam.shape", pcl_cam.shape)
        print("R_wc.shape", R_wc.shape)
        print("T_wc.shape", T_wc.shape)
        pcl_world = ( ( R_wc.T @ pcl_cam.T ) - ( R_wc.T @ T_wc ) ).T
        return mask, pcl_world, pcl_cam, pcl_color

    @staticmethod
    def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
        # Full pipeline

        # * 1. rectify the views
        R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
        R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj
        K = view_i["K"]
        images = [ view_i["rgb"], view_j["rgb"] ]
        applicationProperties = ApplicationProperties("application.yml")
        applicationProperties.initializeProperties()
        sfm_obj = sfm.ReconstructionFrom2DImages(applicationProperties)
        keypoints, descriptions = sfm_obj.detect_SIFT_features( images )
        matches = sfm_obj.match_detected_keypoints(  images, keypoints, descriptions)
        K, uncalibrated_1, uncalibrated_2, calibrated_1, calibrated_2 = sfm_obj.compute_calibrated_coordinates( matches, keypoints ,K)
        E_ransac, inlier_matches = sfm_obj.estimate_ransac( images, matches, keypoints, calibrated_1, calibrated_2)
        transform_candidates = sfm_obj.estimate_pose( E_ransac )
        P1, P2, T, R = sfm_obj.compute_reconstruction( transform_candidates, calibrated_1, calibrated_2 )
        # R, T = sfm_obj.recover_camera_pose( E_ransac, K, calibrated_1, calibrated_2 )
        print("R candidates", R)
        print("T candidates", T)
        R_ji, T_ji, B = TwoViewStereo.compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
        print("R_ji", R_ji)
        print("T_ji", T_ji)
        assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"
        EPS = 1e-6
        R_irect = TwoViewStereo.compute_rectification_R(EPS, T_ji)
        print("R_irect", R_irect)
        rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = TwoViewStereo.rectify_2view(
            view_i["rgb"],
            view_j["rgb"],
            R_irect,
            R_irect @ R_ji,
            view_i["K"],
            view_j["K"],
            u_padding=20,
            v_padding=20,
        )

        # * 2. compute disparity
        assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
        assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
        assert (
            rgb_i_rect.shape == rgb_j_rect.shape
        ), "This hw makes rectified two views to have the same shape"
        disp_map, consistency_mask = TwoViewStereo.compute_disparity_map(
            rgb_i_rect,
            rgb_j_rect,
            d0=K_j_corr[1, 2] - K_i_corr[1, 2],
            k_size=k_size,
            kernel_func=kernel_func,
        )

        # * 3. compute depth map and filter them
        dep_map, xyz_cam = TwoViewStereo.compute_dep_and_pcl(disp_map, B, K_i_corr)

        mask, pcl_world, pcl_cam, pcl_color = TwoViewStereo.postprocess(
            dep_map,
            rgb_i_rect,
            xyz_cam,
            R_wc=R_irect @ R_wi,
            T_wc=R_irect @ T_wi,
            consistency_mask=consistency_mask,
            z_near=0.5,
            z_far=0.6,
        )

        return pcl_world, pcl_color, disp_map, dep_map