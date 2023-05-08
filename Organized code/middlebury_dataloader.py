import numpy as np
import os.path as osp
import os
from tqdm import tqdm
import imageio

class middlebury_DataLoader(object):
    
    @staticmethod
    def load_middlebury_data( datadir , keyword):
        """
        "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"
            The projection matrix for that image is given by K*[R t]
        """
        BBox = {
            "dino": np.array([[-0.041897, 0.001126, -0.037845], [0.030897, 0.088227, 0.035495]]),
            "templeRing": np.array([[-0.023121, -0.038009, -0.091940], [0.078626, 0.121636, -0.017395]])
            }

        try:
            camera_parameter = [osp.join(datadir, keyword + "_par.txt")]
        except:
            print("camera not found or duplicated")
        
        try:
            postition_data = [osp.join(datadir, keyword + "_ang.txt")]
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
                    "rgb": image,
                }
            )
        if len(Image_List) != n_views:
            print("Image_List length not equal to n_views")
        return Image_List