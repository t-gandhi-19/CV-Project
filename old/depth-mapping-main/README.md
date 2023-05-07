# 3D Reconstruction with Stereo Image Pairs using Depth Maps
Creating a depth map and 3D point cloud of stereo image pairs from the Middlebury Stereo Dataset (https://vision.middlebury.edu/stereo/data/)

## Introduction
Stereoscopic vision in humans relates to the ability to get depth information based on the perception of a scene from two different vantage points, one 
from each eye. Our brain processes the shift in the horizontal positions of objects in the views provided from each eye in the visual cortex  giving us
depth perception. We can replicate this task with cameras and some computational processing. Using a picture of the same object or scene from two different 
vantage points, we can compute the distance parts of the scene have shifted and infer depth. The underlying basis of this process is that objects closer to 
the camera eye will travel more in their movement from one image to the next, while objects further from the camera eye will travel less.

## Stereo Image Rectification

In `stereo_rectification.py` you will see the process of rectifying 2 uncalibrated images of a scene into a stereo image pair. (We call 2 images with purely
horizontal shift in the camera planes used to capture them a stereo image pair.) This process involves:

1. Feature detection and matching between two images
2. [Fundamental Matrix](https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)) inference using the [7-point algorithm](https://www.cs.unc.edu/~marc/tutorial/node55.html) 
to identify [epipolar lines](https://en.wikipedia.org/wiki/Epipolar_geometry)
3. Warp each image in the pair with the homography transform needed to make epipolar lines in the image horizontal composited with the translation 
matrix needed to bring images in positive xy space.
4. Violá! Stereo Image Pair Accomplished

*Note: this image pair may not be the best for 3D reconstruction and depth map creation, calibrated camera data like the Middlebury Stereo Dataset provides is much better for nicer 3D reconstruction results!*

## Depth Map Creation

This process can be found in `disparity_map.py`

1. Use block matching to identify matching blocks in the stereo image pair and measure the distance between the center 'x' (columnwise) pixel in each. Store this value in your disparity map.
To classify blocks as matching, use some distance metric like Sum of Absolute Diffences (see below)

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathrm{SSD}(I_1,I_2)&space;=&space;\sum_{x,y&space;\in&space;W}&space;(I_1(x,y)&space;-&space;I_2(x&plus;d,y)&space;)&space;^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathrm{SSD}(I_1,I_2)&space;=&space;\sum_{x,y&space;\in&space;W}&space;(I_1(x,y)&space;-&space;I_2(x&plus;d,y)&space;)&space;^2" title="\mathrm{SSD}(I_1,I_2) = \sum_{x,y \in W} (I_1(x,y) - I_2(x+d,y) ) ^2" /></a>

2. Create disparity maps for many colorspaces and average them to reduce possible inaccuracies
3. Use the formula 

<a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;\frac{ft}{x_l-x_r}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?z&space;=&space;\frac{ft}{x_l-x_r}," title="z = \frac{ft}{x_l-x_r}," /></a>

where f is focal length and t is baseline (or distance between camera centers) to convert disparity to depth

## Post-processing: Filtering the depth map!

This described process can be found in `clean_up_disparity.py`

1. Stack the edges of your disparity map with the edges of an image in your stereo pair
2. Traverse the edge image with varying window sizes
3. At the largest window size containing no edges, store the median of the values in your depth map falling under this window in your filtered depth map

## 3D Point Clouds: Open3D

This process can be found in `disparity_map.py`, specifically in the `display_depth_map` function.


1. Create RGBD matrix with 1 of the 2 images forming your stereo pair and the depth map you have created

    in Open3D 0.7.0  this is 
    `rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(img, depth)`

    in Open3D 0.8.0 this is
    `rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)`

2. Create object storing camera intrinsic values with [PinholeCameraIntrinsic](http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html) using camera focal length and center
3. Create a point cloud with the RGBD image + PinholeCameraIntrinsic object

    in Open3D 0.7.0  this is 
    `pcd_from_depth_map = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, o3d_pinhole)`

    in Open3D 0.8.0 this is
    `pcd_from_depth_map = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pinhole)`

Some notes:

1. If you see any errors related to Open3D in regards to creating an RGBD image or Point Cloud, check your version and use the line of code required for your installed version.
2. `o3d.visualization.draw_geometries` does not work on Mac, I have only confirmed it works on Debian Linux

## Results:

![](https://github.com/cranberrymuffin/depth-mapping/blob/main/results/large_bowling.gif)
![](https://github.com/cranberrymuffin/depth-mapping/blob/main/results/tsukuba_large.gif)
![](https://github.com/cranberrymuffin/depth-mapping/blob/main/results/large_midd.gif)
![](https://github.com/cranberrymuffin/depth-mapping/blob/main/results/large_lampshade.gif)
![](https://github.com/cranberrymuffin/depth-mapping/blob/main/results/large_flowerpots.gif)
