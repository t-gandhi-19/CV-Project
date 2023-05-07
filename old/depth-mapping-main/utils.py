import numpy as np

def get_block(img, y, x, half_window_size):
	'''
	Parameters:
		img1- 3 channel (row, col, colors) numpy array representing a picture
		x and y- coordinates of center pixel or block
		half_window_size-- half the size of the desired block
	Returns:
		get_block -- gets the block of (half_window_size * 2 + 1) centered at y, x
	'''
	row_start = y - half_window_size
	row_end = y + half_window_size + 1

	col_start = x - half_window_size
	col_end = x + half_window_size + 1

	return np.array(img[row_start:row_end, col_start:col_end])