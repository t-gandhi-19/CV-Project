import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
from scipy.stats import mode
import numpy as np
from utils import get_block


def median_filter_block(block):
	'''
	Parameters:
		block-- a block of an image
	Returns:
		a block of the same size where all pixels are filled with the same number (the most frequently appearing color)
	'''
	block[:,:] = mode(block, axis=None)[0]
	return block

def edge_aware_mode_filter(image, edges, mask_set, window_size):
	if(window_size==0 or np.all(mask_set == True)):
		return image;
	[h, w] = image.shape
	half_window_size = int(window_size/2)

	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
	
			edge_block = get_block(edges, y, x, half_window_size)
			
			if(np.any(edge_block==1) or mask_set[y, x] == True):
				continue;

			img_block = get_block(image, y, x, half_window_size)
			image[y, x] = np.median(img_block)
			mask_set[y, x] = True

	return edge_aware_mode_filter(image, edges, mask_set, window_size - 1)

def filter_map(disparity_map, visible_disparity_map, left, window_size):
	'''
	Parameters:
		disparity_map-- disparity map to clean up 
		left-- left stereo image used to make the depth map
	Returns:
		a median filtered disparity map
	'''

	half_window_size = int(window_size/2)
	shape = left.shape
	h = shape[0]
	w = shape[1]

	edges_disparity = feature.canny(visible_disparity_map, sigma=3)
	edges_left = feature.canny(left, sigma=3)
	edges_left[edges_disparity==1] = 1
	
	filtered_map = edge_aware_mode_filter(disparity_map, edges_left, np.zeros(edges_left.shape), window_size)

	return filtered_map