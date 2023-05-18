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

def best_block_dist(block , block_cord , img , search_size , window_size):
    [y , x] = block_cord
    best_sum_sqr_diff = float("inf")
    best_x = x
    best_block  = block_cord
    window_size = int(window_size/2)
    for temp_x in range(max(window_size, x - search_size) , min(img.shape[1] - window_size , x + search_size)):
        temp_block = np.array(img[y - window_size : y + window_size + 1 , temp_x - window_size: temp_x + window_size+ 1])
        sum_sqr_diff = np.sum(np.abs(np.subtract(block , temp_block)))
        if sum_sqr_diff < best_sum_sqr_diff:
            best_sum_sqr_diff = sum_sqr_diff
            best_x = temp_x
            best_block = temp_block
    dist = np.abs(best_x - x)
    return dist

def disp_calculator(l_img , r_img , color_model , window_size , search_size):
    print("hello_world")
    l_img = ing_to_color(l_img , color_model)
    r_img = ing_to_color(r_img , color_model)
    image_height , image_width = l_img.shape[0] , l_img.shape[1]
    # print(image_height , image_width)
    disparity = np.zeros((image_height , image_width))
    w_size = int(window_size/2)
    for i in range(w_size, image_height - w_size):
        for j in range(w_size , image_width - w_size):
            print("calculating for " , i , j)
            block = np.array(l_img[i - w_size: i + w_size + 1 , j - w_size : j + w_size + 1])
            distance = float(best_block_dist(block , [i , j] , r_img , search_size , window_size))
            disparity[i , j] = distance
    
    disparity = disparity[w_size: image_height - w_size , w_size : image_width - w_size]
    return disparity

def get_disparity_map(cfg , l_img , r_img):
    print("hello_world")
    color_models = cfg.TASK.COLOR_MODELS
    print(color_models)
    disparity_map_dict = {}
    for color_model in color_models:
        print(color_model)
        disparity_map_dict[color_model] = disp_calculator(l_img, r_img, color_model , window_size = cfg.TASK.WINDOW_SIZE , search_size = cfg.TASK.SEARCH_SIZE)

    Avg_disparity_map = np.zeros(disparity_map_dict[color_models[0]].shape)
    for color_model in color_models:
        print(color_model)
        Avg_disparity_map += disparity_map_dict[color_model]
    Avg_disparity_map /= len(color_models)
    Visible_disparity_map = Avg_disparity_map.astype(np.float64) / cfg.TASK.SEARCH_SIZE
    Visible_disparity_map = 255 * Visible_disparity_map
    Visible_disparity_map = Visible_disparity_map.astype(np.uint8)
    return Visible_disparity_map

def get_disparity_map(cfg , l_img , r_img):
    print("hello_world")
    color_models = cfg.TASK.COLOR_MODELS
    print(color_models)
    disparity_map_dict = {}
    for color_model in color_models:
        print(color_model)
        disparity_map_dict[color_model] = disp_calculator(l_img, r_img, color_model , window_size = cfg.TASK.WINDOW_SIZE , search_size = cfg.TASK.SEARCH_SIZE)

    Avg_disparity_map = np.zeros(disparity_map_dict[color_models[0]].shape)
    for color_model in color_models:
        print(color_model)
        Avg_disparity_map += disparity_map_dict[color_model]
    Avg_disparity_map /= len(color_models)
    Visible_disparity_map = Avg_disparity_map.astype(np.float64) / cfg.TASK.SEARCH_SIZE
    Visible_disparity_map = 255 * Visible_disparity_map
    Visible_disparity_map = Visible_disparity_map.astype(np.uint8)
    return Visible_disparity_map


def get_depth_map(cfg , disparity_map , K1 , K2 , baseline , doffs):
    depth_map = np.zeros(disparity_map.shape)
    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            if disparity_map[i , j] != 0:
                depth_map[i , j] = (K1 * baseline) / (disparity_map[i , j] + K2 + doffs)
    return depth_map