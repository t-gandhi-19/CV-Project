import cv2
import numpy as np

def ing_to_color(image , color_model):
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
	im_yuv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2YUV)

    if color_model == "gray":
        return cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    elif color_model == "hsv":
        return cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
    elif color_model == "lab":
        return cv2.cvtColor(image , cv2.COLOR_BGR2LAB)
    elif color_model == "yuv":
        return cv2.cvtColor(image , cv2.COLOR_BGR2YUV)
    elif color_model == "xyz":
        return cv2.cvtColor(image , cv2.COLOR_BGR2XYZ)
    elif color_model == "ycrcb":
        return cv2.cvtColor(image , cv2.COLOR_BGR2YCrCb)
    elif color_model == "r":
        return im_bgr[:,:,2]
    elif color_model == "g":
        return im_bgr[:,:,1]
    elif color_model == "b":
        return im_bgr[:,:,0]
    elif color_model == "l":
        return im_lab[:,:,0]
    elif color_model == "y":
        return im_yuv[:,:,0]
    elif color_model == "rgb":
        return cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    else:
        print("Invalid color model!")
        exit(0)

def disp_calculator(l_img , r_img , color_model , window_size , search_size):
    print("hello_world")
    l_img = ing_to_color(l_img , color_model)
    r_img = ing_to_color(r_img , color_model)
    image_height , image_width = l_img.shape
    disparity = np.zeros((image_height , image_width))

def get_disparity_map(cfg , l_img , r_img):
    print("hello_world")
    color_models = cfg.COLOR_MODELS
    disparity_map_dict = {}
    for color_model in color_models:
        disparity_map_dict[color_model] = disp_calculator(l_img, r_img, color_model , window_size = cfg.TASK.WINDOW_SIZE , search_size = cfg.TASK.SEARCH_SIZE)
