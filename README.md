

### Advanced Lane Finding:

The goal of this project is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.


The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Instructions:

* The images for camera calibration are stored in the folder called camera_cal. The images in test_images are for testing the pipeline on single frames. To extract more test images from the videos, you can simply use an image writing method like cv2.imwrite(), i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.

* The examples of the output from each stage of the pipeline saved in the folder called output_images, and included a description in this writeup for the project of what each image shows. The video called project_video.mp4 is the video your pipeline should work well on.

* The challenge_video.mp4 video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions. The harder_challenge.mp4 video is another optional challenge and is brutal!

[//]: # (Image References)




## Rubric Points

##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

1.  Compute the camera calibration matrix and distortion coefficients given a set of chessboard images: To calculate camera calibration for Images that are stored in the folder called camera_cal, I compute the camera matrix and distortion co-efficients to undistort the all images. I used the calibration_calculate, cal_distortion functionsto calulate calibration matrix and distortion coefficients and then by using undistortion functions (in the corners_unwarp functions) i undistorted all images and by using  the corners_unwarp function I transformed them to Front view. An example of a distortion-corrected image is presented below:
<p align="right">
<img src="./output_images/1.png" alt="compute the camera calibration" />
<p align="right">



2.  Gradients and color thresholds,Sobel X , Y , Magnitude and Direction Gradients: 

*   First I applied thresholds on X, Y (abs_sobel_thresh function), magnitude (mag_thresh function) and direction (dir_threshold function) gradients to combine them into a binaray image(named combined_gradient) by using combined_thresholds function.
<p align="right">
<img src="./output_images/2.png" alt=" Gradients and color thresholds" />
<p align="right">
 <p align="right">
<img src="./output_images/3.png" alt=" Gradients and color thresholds" />
<p align="right">
 
 *  Then I combine the binary image from the last step (combined_gradient)  with  of the threshold color channel H from HLS color spaces (HLScolor function) to obtain a binary image (combined_thresholds_color1) and then use this binary image to combine with the thresol L channel from LUV space color (LUVcolor function) to get another binary image (combined_thresholds_color2) and then use the resulted binary image from the last step to combine with the threshold R channel from RGB (RGBcolor function)and the resulted image (combined_thresholds_color3) from last step with the threshol L channel from LAB colorsystem to obtain the binary threshold image 4 (combined_thresholds_color4).
 
 <p align="right">
<img src="./output_images/4.png" alt=" Gradients and color thresholds" />
<p align="right">
 <p align="right">
<img src="./output_images/5.png" alt=" Gradients and color thresholds" />
<p align="right">
 <p align="right">
<img src="./output_images/6.png" alt=" Gradients and color thresholds" />
<p align="right">
 <p align="right">
<img src="./output_images/7.png" alt=" Gradients and color thresholds" />
<p align="right">



3. Perspective transform ("birds-eye view"): First, I extracted the source and distinction points to perform a perspective transformation with help of the calc_warp_points function, then I feed the binary threshold image from the last step into the transform_image() function to get a bird's eye view from above which will be rectified by adding morphological dilation and erosion to make the edge lines continuous. 

 <p align="right">
<img src="./output_images/8.png" alt="  Perspective transform" />
<p align="right">
 <p align="right">
<img src="./output_images/9.png" alt="  Perspective transform" />
<p align="right">

4.  Nois Detection: In this step Noise will be detected by using a function named noise_detect and if the result of this function is True, instead of using the combined_thresholds_color4 variable wir are goining to use the variable combined_thresholds_color1 which has better result in noisy images for Perspective transforming (step 3).


5. Implementing of sliding Windows and Fit a Polynomial: In order to detect the lane pixels from the warped image, First, a histogram of the lower half of the warped image is created by using the get_histogram function then the starting left and right lanes positions are selected by looking to the max value of the histogram to the left and the right of the histogram's mid position.
Second a technique known as Sliding Window is used to identify the most likely coordinates of the lane lines in a window to found x & y coordinates of non-zero pixels.
which slides vertically through the image for both the left and right line.
Finally, usign the coordinates previously calculated, a second order polynomial is calculated for both the left and right lane line(Numpy's function np.polyfit will be used to calculate the polynomials) and the track lines are drawn.



 <p align="right">
<img src="./output_images/10.png" alt="Implementing of sliding Windows and Fit a Polynomial" />
<p align="right">


6. Finding the Lines: Search from Prior:Since consecutive frames are likely to have lane lines in roughly similar positions it is reasonable to assume that the lines will remain there in future video frames. detect_similar_lines() uses the previosly calculated line_fits to try to identify the lane lines in a consecutive image. If it fails to calculate it, it invokes detect_lines() function to perform a full search.


 <p align="right">
<img src="./output_images/11.png" alt=". Finding the Lines: Search from Prior" />
<p align="right">

7. Determine the curvature of the lane, and vehicle position with respect to center: To calculate the radius of curvature of the lane and the position of the vehicle with respect to center I used the below preented functions:
```python
def curvature_radius (leftx, rightx, img_shape, xm_per_pix=3.7/800, ym_per_pix = 25/720):
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    
    leftx = leftx[::-1] 
    rightx = rightx[::-1] 
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)

```
```python
def car_offset(leftx, rightx, img_shape, xm_per_pix=3.7/800):
    ## Image mid horizontal position 
    mid_imgx = img_shape[1]//2
        
    ## Car position with respect to the lane
    car_pos = (leftx[-1] + rightx[-1])/2
    
    ## Horizontal car offset 
    offsetx = (mid_imgx - car_pos) * xm_per_pix

    return offsetx

```

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
fotos are in jupyter environment.
I used the calibration_calculate and cal_distortion undistortion functions to calulate calibration matrix and distortion coefficients amd by using the corners_unwarp functions i transformed fotos to bird eye view. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I applied thresholds on X, Y and direction und magnitude gradients and combined them with Gradient of color channel H from HLS color spaces to obtain a binary thresholded image.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

First, I extracted the source and distinction points to perform a perspective transformation. Then I feed the binary threshold image from the last step into the transform_image(img) function to get a bird's eye view from above.

```python

    leftupperpoint  = [568,470]
    rightupperpoint = [717,470]
    leftlowerpoint  = [260,680]
    rightlowerpoint = [1043,680]

    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])
    M_inv = cv2.getPerspectiveTransform(dst, src)
```




#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

On this level I perform a sliding window search, startingwith the base likely positions of the 2 lane (shown in the first image), calculated from the histogram. I used 9 windows with a width of 100 pixels. The x & y coordinates of non-zero pixels are found, a polynomial is adjusted for these coordinates and the track lines are drawn.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did  this  in the function`measure_curvature_real`
```python
def measure_curvature_real(img_shape,left_fitx,right_fitx):
    ploty = np.linspace(0, img_shape[0]-1, num=img.shape[0])# to cover same y-range as image
    ym_per_pix = 30/img_shape[0] # meters per pixel in y dimension
    xm_per_pix = 3.7/700 
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    

    left_curverad = ((1+ (2* left_fit_cr[0]*ym_per_pix* y_eval+left_fit_cr[1]) **2) **1.5)/np.absolute(2*left_fit_cr[0])  ## Implement         the calculation of the left line here

right_curverad =((1+ (2* right_fit_cr[0]*ym_per_pix* y_eval+right_fit_cr[1]) **2) **1.5)/np.absolute(2*right_fit_cr[0])      Implement the calculation of the right line here

    
    h = ym_per_pix
    car_position = img_shape[1]/2
    l_fit_x_int = left_fit_cr[0]*h**2 + left_fit_cr[1]*h + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*h**2 + right_fit_cr[1]*h + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center_dist = (car_position - lane_center_position) * xm_per_pix

    return left_curverad, right_curverad,center_dist

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `Drawing()`.  an example of my result is ai the jupyter notebook.



---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline is applied to videos

1-undistor image.

2-claculate x gradient.

3-calculate y gradient.

4-calculate direction gradient.

5-calculate magnitude gradient.

6-calculate color threshold of S channel from HLS color space.

7-calculate color threshold of L channel from LUV color space.

8-calculate color threshold of B channel from LAB color space.

9- combined all gradients and thresholds fom last steps.

10-Feed the threshold binary image to tensform perspectice function to get a bird eye view.

11-add morphological dilation and erosion to make the edge lines continuous

12-perform nois detection function and if the answer of it is True we can take the binary image from combined_thresholds_color1 that has better result in noisy images.

13-perform fit polynomial algorithm.

14-perform search around poly algorithm.



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Take a better perspective transform: choose a smaller section to take the transform since this video has sharper turns and the lenght of a lane is shorter than the previous videos.
