import numpy as np
import os.path as osp
import os
from tqdm import tqdm
import imageio
from pypfm import PFMLoader

class middlebury_DataLoader(object):
    
    @staticmethod
    def get_parameters(path_to_calibration_file):
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

    @staticmethod
    def load_middlebury_data3d( datadir , keyword):
        """
        "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"
            The projection matrix for that image is given by K*[R t]
        """
        BBox = {
            "dino": np.array([[-0.041897, 0.001126, -0.037845], [0.030897, 0.088227, 0.035495]]),
            "templeRing": np.array([[-0.023121, -0.038009, -0.091940], [0.078626, 0.121636, -0.017395]])
            }

        try:
            camera_parameter = osp.join(datadir, keyword + "_par.txt")
        except:
            print("camera not found or duplicated")
        
        try:
            postition_data = osp.join(datadir, keyword + "_ang.txt")
        except:
            print("camera not found or duplicated")
        
        with open(camera_parameter) as f:
            cam_parameter_data = f.readlines()

        with open(postition_data) as f:
            ang_parameter_data = f.readlines()

        Image_List = []
        n_views = int(cam_parameter_data.pop(0))

        for cam, ang in tqdm( zip( cam_parameter_data, ang_parameter_data ) ):
            cam_split = cam.split(" ")
            image_filename = cam_split[0]
            image = imageio.imread( osp.join( datadir, image_filename ) )
            cam_split = cam_split[1:]
            K = np.array(cam_split[:9]).reshape(3, 3).astype(np.float64)
            R = np.array(cam_split[9:18]).reshape(3, 3).astype(np.float64)
            t = np.array(cam_split[18:]).astype(np.float64)
            ang_split = ang.split(" ")
            lat = float(ang_split[0])
            lon = float(ang_split[1])
            Image_List.append(
                {
                    "K": K.astype(np.float64),
                    "R": R.astype(np.float64),
                    "T": t.astype(np.float64),
                    "lat": lat,
                    "lon": lon,
                    "Image": image,
                }
            )
        if len(Image_List) != n_views:
            print("Image_List length not equal to n_views")
        return Image_List
    
    @staticmethod
    def load_middlebury_data2d(datadir):
        list_files = os.listdir(datadir)
        # remove .DS_Store
        list_files = [x for x in list_files if x != ".DS_Store"]
        list_files.sort()
        Image_List = []
        for file in list_files:
            print(file)
            left_image = imageio.imread(osp.join(datadir, file, "im0.png"))
            right_image = imageio.imread(osp.join(datadir, file, "im1.png"))
            loader = PFMLoader(color=False, compress=False)
            disp0 = loader.load_pfm(osp.join(datadir, file, "disp0.pfm"))
            disp1 = loader.load_pfm(osp.join(datadir, file, "disp1.pfm"))
            K_left , K_right , doffs , baseline = middlebury_DataLoader.get_parameters(osp.join(datadir, file, "calib.txt"))
            # reshape  K_left and K_right TO 3x3 FROM 1x9
            K_left = K_left.reshape(3,3)
            K_right = K_right.reshape(3,3)
            Image_List.append(
                {
                    "left_image": left_image,
                    "right_image": right_image,
                    "disp0": disp0,
                    "disp1": disp1,
                    "K_left": K_left,
                    "K_right": K_right,
                    "doffs": doffs,
                    "baseline": baseline,
                }
            )
        return Image_List