
# Self-Driving Car Engineer Nanodegree
# Advanced Lane Lines

In this project we will revisit the lane line detection project using some more advanced computer vision techniques.


```python
# Import packages.
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from moviepy.editor import VideoFileClip
%matplotlib inline
```

## Camera Calibration

Before we begin detecting lane lines, we will need to calibrate the camera. This will allow us to correct for distortions in the images taken by the camera.

Camera distortion will be determined by looking for corners in a chessboard image taken from various angles. Loop through each of these calibration images to construct a mapping from image coordinates to real world coordinates.


```python
# Prepare object points.
nx = 9 # The number of horizontal inside corners.
ny = 6 # The number of vertical inside corners.

# Arrays to store detected image points and object points.
objpoints = [] # 3D points in real world space.
imgpoints = [] # 2D points in image space.

# Prepare the object points. These will be the same for all images.
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Loop through the calibration images/
nb_images = 0
nb_detected = 0
for filename in os.listdir('camera_cal'):
    nb_images += 1
    # Load the calibration image.
    img = cv2.imread('camera_cal/' + filename)
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners.
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        nb_detected += 1
        # Add object points and image points.
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw detected corners on the image.
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    # Save the images, with corners annotated if found.
    cv2.imwrite('output_images/corners-' + filename, img)

# Print results of corner detection.
print('Successfully found corners in {} of {} calibration images.'.format(nb_detected, nb_images))
```

    Successfully found corners in 17 of 20 calibration images.


## Distortion Correction

Now that the mappings from image space to the real world have been collected, we can use OpenCV to calculate the camer matrix and distortion coefficients. We can verify these values by undistorting the calibration images.


```python
# Get the shape of the calibration images.
img = cv2.imread('camera_cal/calibration1.jpg')
shape = img.shape[0:2]

# Calculate the camera matrix and distortion coefficients.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)  

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

img = cv2.imread('camera_cal/calibration1.jpg')
dst = undistort(img)

# Plot the original and undistorted images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
```




    <matplotlib.text.Text at 0x115d96c88>




![png](output_5_1.png)



```python
# Loop through the calibration images.
for filename in os.listdir('camera_cal'):
    # Load the calibration image.
    img = cv2.imread('camera_cal/' + filename)
    # Undistort the image.
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # Save the undistorted images.
    cv2.imwrite('output_images/undistorted-' + filename, img)
```

## Color/Gradient Thresholds

With distortion correction now solved, we can move on to work more directly with finding lanes. The next step will be to apply some mixture of color and gradient thresholding to identify points that likely belong to the lane line.

### HLS Saturation Thresholds

The BGR color representation that OpenCV uses natively to process images does not lend itself well to lane detection. Instead, we will convert the images into an alternative color space that is more robust to road conditions. In particular, the saturation channel of the HSL color space has shown itself to be particularly good at identifying both white and yellow lane lines under various lighting conditions.


```python
SATURATION_THRESHOLD = (100, 255)

def saturation_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    binary = np.zeros_like(s)
    binary[(s > SATURATION_THRESHOLD[0]) & (s <= SATURATION_THRESHOLD[1])] = 1
    return binary

img = cv2.imread('test_images/test1.jpg')
img = undistort(img)
dst = saturation_threshold(img)

# Plot the original and thresholded images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Saturation Threshold', fontsize=30)
```




    <matplotlib.text.Text at 0x10f8c7d68>




![png](output_8_1.png)


Thresholding on the saturation channel in the HLS colorspace has done a good job of isolating both the yellow and white lane lines, but it has worked better in some of the test images than others.


```python
# Loop through the test images and apply the saturation threshold.
for filename in os.listdir('test_images'):
    img = cv2.imread('test_images/' + filename)
    img = undistort(img)
    img = saturation_threshold(img)
    # Convert from binary to grayscale before saving.
    img *= 255
    cv2.imwrite('output_images/saturation-threshold-' + filename, img)
```

### Sobel Gradient Thresholds

In the first version of this project, we used Canny edge detection to isolate lanes lines. This worked well, but the algorithm found edges in both the horizontal and vertical directions. Since lane lines are mostly vertical, it would be better to emphasize the vertical edges. This is possible through the use of Sobel filters.


```python
SOBEL_X_THRESHOLD = (75, 200)
SOBEL_Y_THRESHOLD = (75, 200)

def sobel_threshold(img, dir='x', thresh=SOBEL_X_THRESHOLD):
    if dir == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255. * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary

def combined_sobel(img):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Sobel thresholds.
    sobel_x = sobel_threshold(gray, 'x', SOBEL_X_THRESHOLD)
    sobel_y = sobel_threshold(gray, 'y', SOBEL_Y_THRESHOLD)
    # Combine the filters.
    binary = cv2.bitwise_or(sobel_x, sobel_y)
    return binary

img = cv2.imread('test_images/test1.jpg')
img = undistort(img)
dst = combined_sobel(img)

# Plot the original and thresholded images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Sobel Threshold', fontsize=30)
```




    <matplotlib.text.Text at 0x110693828>




![png](output_12_1.png)



```python
# Loop through the test images and apply the sobel threshold.
for filename in os.listdir('test_images'):
    img = cv2.imread('test_images/' + filename)
    img = undistort(img)
    img = combined_sobel(img)
    # Convert from binary to grayscale before saving.
    img *= 255
    cv2.imwrite('output_images/combined-sobel-' + filename, img)
```

The Sobel threshold filter didn't work as well with the washed out lines in this image as the color saturation filter, but it did work better on other test images.

### Combined Thresholds

The saturation and sobel filters each performed differently based on various conditions of the images. Rather than relying on one or the other, we can combine the filters to produce a thresholding function that takes advantages of the unique properties of each.


```python
def combined_threshold(img):
    saturation_binary = saturation_threshold(img)
    sobel_binary = combined_sobel(img)
    binary = cv2.bitwise_or(saturation_binary, sobel_binary)
    return binary

img = cv2.imread('test_images/test1.jpg')
img = undistort(img)
dst = combined_threshold(img)

# Plot the original and thresholded images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Combined Threshold', fontsize=30)
```




    <matplotlib.text.Text at 0x11cf449e8>




![png](output_15_1.png)


By combining color based and gradient based thresholding techniques, we were able to isolate lane lines in images where neither technique alone would have been sufficient. While we could continue to try different types and combinations of thresholds, using just these two will give us a good start.


```python
# Loop through the test images amd apply the combined threshold.
for filename in os.listdir('test_images'):
    img = cv2.imread('test_images/' + filename)
    img = undistort(img)
    img = combined_threshold(img)
    # Convert from binary to grayscale before saving.
    img *= 255
    cv2.imwrite('output_images/combined-threshold-' + filename, img)
```

## Perspective Transform

To simplify the detection of lane lines as well as to compute the radius of curvature, we will create a perspective transform that will warp the images so that they appear as if we were looking at them from above.

The test images all feature various degrees of curving roads, which makes it hard to identify points that can be used to define the transform. Instead of using the provided test images, we scanned through the project video to find a relatively straight section of road to identify source points for the transform. This new image has been saved to the ```test_images``` directory as ```straight.jpg```.


```python
PERSPECTIVE_SRC_POINTS = np.float32([[250, 719], [505, 520], [820, 520], [1180, 719]])
PERSPECTIVE_DST_POINTS = np.float32([[250, 719], [250, 520], [1180, 520], [1180, 719]])
M = cv2.getPerspectiveTransform(PERSPECTIVE_SRC_POINTS, PERSPECTIVE_DST_POINTS)
Minv = cv2.getPerspectiveTransform(PERSPECTIVE_DST_POINTS, PERSPECTIVE_SRC_POINTS)

def warp(img):
    image_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)

def unwarp(img):
    image_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, Minv, image_size, flags=cv2.INTER_LINEAR)

img = cv2.imread('test_images/test1.jpg')
img = undistort(img)
img = combined_threshold(img)
dst = warp(img)

# Plot the original and warped images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(img, cmap='gray')
ax1.set_title('Combined Threshold', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Birds-Eye Image', fontsize=30)
```




    <matplotlib.text.Text at 0x118c8db00>




![png](output_19_1.png)



```python
# Loop through the test images amd apply the perspective transform.
for filename in os.listdir('test_images'):
    img = cv2.imread('test_images/' + filename)
    img = undistort(img)
    img = combined_threshold(img)
    img = warp(img)
    # Convert from binary to grayscale before saving.
    img *= 255
    cv2.imwrite('output_images/warp-' + filename, img)
```

## Finding Lane Lines

### Histogram

To first detect a starting point for the lane lines, we will create a histogram of possible points in the lower half of the warped image. The points in the lower half of the screen haven't been as distorted by the transform to the birds-eye view and are more likely to be closer to vertical, which will help keep the width of the histogram peaks small and better localized.


```python
def histogram(img):
    return np.sum(img[img.shape[0]/2:,:], axis=0)

img = cv2.imread('test_images/test1.jpg')
img = undistort(img)
img = combined_threshold(img)
img = warp(img)
hist = histogram(img)

# Plot the original and histogram images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(img, cmap='gray')
ax1.set_title('Birds-Eye Image', fontsize=30)
ax2.plot(hist)
ax2.set_title('Histogram', fontsize=30)
```




    <matplotlib.text.Text at 0x11a92c668>




![png](output_22_1.png)



```python
# Loop through the test images amd apply the perspective transform.
for filename in os.listdir('test_images'):
    img = cv2.imread('test_images/' + filename)
    img = undistort(img)
    img = warp(img)
    img = combined_threshold(img)
    hist = histogram(img)
    plt.title('Histogram')
    plt.plot(hist)
    plt.savefig('output_images/histogram-' + filename, bbox_inches='tight')
    plt.close()
```

### Detecting Lane Lines

We have now applied all of the thresholding, transformations, and histograms and are ready to try to detect the lane lines. We will use a utility class to track the progress of the detection of each lane and pass that information along to help speed up detection in subsequent frames.


```python
# Dimensions of sliding window to search for lane points.
SLIDING_WINDOW_WIDTH = 200
SLIDING_WINDOW_HEIGHT = 90

# Conversion of pixels to meters, as given in the lecture notes.
X_METERS_PER_PIXEL = 3.7 / 700.
Y_METERS_PER_PIXEL = 30. / 720.

# Number of frames to smooth over.
N_FRAMES = 3

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        #polynomial coefficients for the last n iterations
        self.previous_fits = []  
        #polynomial coefficients for the most recent fit
        self.current_fit = None  
        #radius of curvature of the line in meters
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
def fit_point(y, line):
    # The lane fit is in meters, so we need to convert from pixel
    # values to meters and then back again.
    ym = y * Y_METERS_PER_PIXEL
    xm = line.current_fit[2] + line.current_fit[1]*ym + line.current_fit[0]*ym*ym
    return int(xm / X_METERS_PER_PIXEL)
        
def detect_lane(img, line, xmin, xmax):
    if line.detected:
        # A line was detected in a previous frame, so we can start
        # search for the new line within a sliding window around the
        # old one.
        xstart = fit_point(img.shape[0] - 1, line)
    else:
        # If we haven't found a lane line yet, start with the peak
        # of the histogram as the best guess.
        hist = histogram(img)
        xstart = xmin + np.argmax(hist[xmin:xmax])
    
    # Process sliding windows of points.
    xleft = xstart - SLIDING_WINDOW_WIDTH // 2
    ybottom = img.shape[0]
    allx = []
    ally = []
    while ybottom > 0:
        # clip the sliding window start to the bounds of the image.
        xleft = min(max(0, xleft), img.shape[1])
        xright = min(xleft + SLIDING_WINDOW_WIDTH, img.shape[1])
        ytop = max(0, ybottom - SLIDING_WINDOW_HEIGHT)

        # Loop through every pixel in the sliding window to see if
        # it should be added to the candidate line.
        xvalues = []
        for y in range(ytop, ybottom):
            for x in range(xleft, xright):
                if img[y][x] > 0:
                    xvalues.append(x)
                    allx.append(x)
                    ally.append(y)

        # Move the sliding window up and recenter it on the mean
        # x value of the detected points.
        ybottom -= SLIDING_WINDOW_HEIGHT
        if len(xvalues) > 0:
            xleft = int(np.mean(xvalues) - SLIDING_WINDOW_WIDTH // 2)
    
    # Check to see if we detected at least a reasonable number of candidate
    # points for the line. Otherwise just reuse the previously fit line and
    # trigger a full scan on the next frame.
    if len(allx) < 10:
        line.detected = False
        line.previous_fits = []
    else:
        # Calculate coefficients and radius of curve fit to detected points.
        allx = np.array(allx)
        ally = np.array(ally)
        coeffs = np.polyfit(ally*Y_METERS_PER_PIXEL, allx*X_METERS_PER_PIXEL, 2)
        
        # Add the newly fit line to the previous lines and the compute the
        # average to use as the smoothed line for this frame.
        line.previous_fits.append(coeffs)
        line.previous_fits = line.previous_fits[-N_FRAMES:]
        coeffs = np.average(line.previous_fits, axis=0)
                
        # Evaluate radius at bottom of image.
        y_eval = (img.shape[0] - 1) * Y_METERS_PER_PIXEL
        radius = ((1 + (2*coeffs[0]*y_eval + coeffs[1])**2)**1.5)/np.absolute(2*coeffs[0])
        # Calculate distance from center
        xline = coeffs[2] + coeffs[1]*y_eval + coeffs[0]*y_eval*y_eval
        xcenter = X_METERS_PER_PIXEL * img.shape[1] / 2.
        base_pos = abs(xline - xcenter)
                
        # Save line parameters.
        line.detected = True
        line.current_fit = coeffs
        line.radius_of_curvature = radius
        line.line_base_pos = base_pos

def detect_lanes(img, left_line, right_line):
    # Find the center point of the image. For now we will assume that the
    # car won't change lanes so that we can look for the left lane in the
    # left half of the image and the right lane in the right half.
    detect_lane(img, left_line, 0, img.shape[1] // 2 - 1)
    detect_lane(img, right_line, img.shape[1] // 2, img.shape[1] - 1)
```

### Draw the Lane Mask

Once we have detected the lanes and fit polynomial equations to their points, we can use those polynomials to fill the space between those curves to show the area covered by the lane on the road. Later we can take this mask and unwarp it back to the camera's point of view and overlay it on the original image of the road.


```python
def draw_mask(img, left_line, right_line):
    channel_zeros = np.zeros_like(img).astype(np.uint8)
    mask = np.dstack((channel_zeros, channel_zeros, channel_zeros))
    for y in range(img.shape[0]):
        xleft = fit_point(y, left_line)
        xright = fit_point(y, right_line)
        cv2.line(mask, (xleft, y), (xright, y), (0, 255, 255))
    return mask

img = cv2.imread('test_images/test1.jpg')
img = undistort(img)
img = combined_threshold(img)
img = warp(img)

left_line = Line()
right_line = Line()
detect_lanes(img, left_line, right_line)
mask = draw_mask(img, left_line, right_line)

# Plot the original and warped images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(img, cmap='gray')
ax1.set_title('Birds-Eye Image', fontsize=30)
ax2.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
ax2.set_title('Lane Mask', fontsize=30)
```




    <matplotlib.text.Text at 0x123d3beb8>




![png](output_27_1.png)


## Image Processing Pipeline

The image processing pipeline will form the basis of the lane lines detection. The pipeline will combine all of the steps developed above to process a single image or frame of video. Once the image has been assembled, it will be annotated with the curvature of the lane and the distance of the car from the center of the lane.


```python
def pipeline(img, left_line, right_line):
    # Undistort the image to correct for camera distortions.
    img = undistort(img)
    # Apply a Gaussian blur to smooth out the image.
    img = cv2.GaussianBlur(img,(3, 3), 0)
    # Apply thresholds to isolate potential lane line points.
    thresh = combined_threshold(img)
    # Warp the perspective to a birds-eye view 
    warped = warp(thresh)
    # Detect the lane lines
    detect_lanes(warped, left_line, right_line)
    # Draw the lane mask.
    warped_mask = draw_mask(warped, left_line, right_line)
    # Unwarp the lane maske
    mask = unwarp(warped_mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # Combine the original image with the lane mask.
    combined = cv2.addWeighted(mask, 0.4, img, 1., 0.)
    # Annotate lane curvature and distance from center.
    # Take the average curvature of the left and right lines.
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
    combined = cv2.putText(combined, 'Curvature: %0.1fm' % curvature, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1., (255, 255, 255))
    # Calculate the car's offset from the center of lane.
    offset = int(100. * (left_line.line_base_pos - right_line.line_base_pos) / 2.)
    combined = cv2.putText(combined, 'Distance from Center: %dcm' % offset, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1., (255, 255, 255))
    # Return the processed image.
    return combined

img = cv2.imread('test_images/test1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = pipeline(img, Line(), Line())

# Plot the original and processed images.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Processed Image', fontsize=30)
```




    <matplotlib.text.Text at 0x123cba9e8>




![png](output_29_1.png)



```python
# Loop through the test images and apply the pipeline.
for filename in os.listdir('test_images'):
    if filename.startswith('test'):
        img = cv2.imread('test_images/' + filename)
        left_line = Line()
        right_line = Line()
        img = pipeline(img, Line(), Line())
        cv2.imwrite('output_images/pipeline-' + filename, img)
```

## Video Pipeline

Now all the steps are in place to detect lines in the videos. First we will generate a callable lambda expression that the movie writer can use to send each image through the image pipeline before writing the processed image to the new video file.


```python
def process_image():
    left_line = Line()
    right_line = Line()
    return (lambda img: pipeline(img, left_line, right_line))

for filename in ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']:
    clip = VideoFileClip(filename)
    out = clip.fl_image(process_image())
    out.write_videofile('processed-' + filename, audio=False)
    
```

    [MoviePy] >>>> Building video processed-project_video.mp4
    [MoviePy] Writing video processed-project_video.mp4


    100%|█████████▉| 1260/1261 [10:28<00:00,  2.18it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: processed-project_video.mp4 
    
    [MoviePy] >>>> Building video processed-challenge_video.mp4
    [MoviePy] Writing video processed-challenge_video.mp4


    100%|██████████| 485/485 [03:55<00:00,  1.96it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: processed-challenge_video.mp4 
    
    [MoviePy] >>>> Building video processed-harder_challenge_video.mp4
    [MoviePy] Writing video processed-harder_challenge_video.mp4


    100%|█████████▉| 1199/1200 [10:19<00:00,  1.87it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: processed-harder_challenge_video.mp4 
    


## Results

The processed videos can be found in the main directory.

* ```processed-project_video.mp4```: The project video.
* ```processed-challenge_video.mp4```: The first challenge video.
* ```processed-harder_challenge_video.mp4```: The harder challenge video.

As can be seen from the videos above, the main project video is able to fairly accurately track the progress of the car. The thresholding and line fitting techniques used, however, are still not sophisticated enough to handle the more challenging videos with their harsher driving conditions.

Intermediate progress images were saved as the pipeline was being developed. They can be examined to see the effect of each transformation as the pipeline progresses. These images are stored in the ```output_images``` directory.

* ```corners-calibration-*.jpg```: The camera calibration images showing the detected corners. Not all of the images were able to have their corners dected.
* ```undistorted-calibration-*.jpg```: The camera calibration images after the undistorting them to remove camera artifacts.
* ```saturation-threshold-*.jpg```: The test images after the saturation threshold has been applied to the HLS representation of the images.
* ```combined-sobel-*.jpg```: The test images after the sobel edge filters have been applied.
* ```combined-threshold-*,jpg```: The test images after the saturation threshold and sobel filter thresholds have been applied.
* ```warp-*.jpg```: The thresholded images after being warped into a birds-eye perspective.
* ```histogram-*.jpg```: A histogram of x values in the lower half of the birds-eye images. These are used to bootstrap the detection of the start of the lane lines.
* ```pipeline-*.jpg```: The final result after apply all of the image transformations to the test images.

The project was considerably more difficult than the first lane detection project. However, the advanced computer vision techniques were able to provide much more information to the self-driving car. These techniques were able to detect curved lanes and and make some estimates on the amount of curvature and how far away the car had drifted from the center of the lane.

Unfortunately, the success of these computer vision technique are highly dependent on selecting the best filters and transformation as well as accurately tuning the various parameters of those algorithms. It would be nice to avoid all of this work by training a deep learning network to find and tune the parameters itself.
