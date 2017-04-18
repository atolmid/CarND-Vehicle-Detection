##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Random_draw_boxes_example.png
[image2]: ./output_images/draw_boxes_around_vehicles.png
[image3]: ./output_images/RGB_histograms.png
[image4]: ./output_images/spatially_binned_pictures.png
[image5]: ./output_images/car_non-car_image_example.png
[image6]: ./output_images/car_hog_example.png
[image7]: ./output_images/non-car_hog_example.png
[image8]: ./output_images/car_non-car_features_example.png
[image9]: ./output_images/find_cars_example.png
[image10]: ./output_images/heatmap1.png
[image11]: ./output_images/heatmap2.png
[image12]: ./output_images/heatmap3.png
[image13]: ./output_images/heatmap4.png
[image14]: ./output_images/heatmap5.png
[image15]: ./output_images/heatmap6.png
[image16]: ./output_images/slide_window.png
[video1]: ./project_output.mp4
[video2]: ./combined_project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

Initial step was to take the source code provided at the lecture exercises, and use it as a basis.
Therefore, the structure of the Vehicle\_detection.ipynb jupyter notebook (that contains the source code and the results), and the various steps correspond to the sequence used during the lectures.

The first thing was to demonstrate that I was able to draw bounding boxes on images.
Therefore, I tried drawing random bounding boxes on a test image.
The result can be seen in Image1:

![alt text][image1]

The next step, was to use the cutout images of cars provided during the lectures, in order to detect vehicles in a test image, and then draw their bounding boxes.
An example is Image2:

![alt text][image2]

Then, I defined two similar functions, in order to compute color histogram features.
Their difference is that the first one, besides the feature vector, returns the individual channel histograms and the bin_center.
This function is used in order be able to plot the individual channel histograms.
Both also normalise the histogram values.
An example can be seen in Image3:

![alt text][image3]

After that, I define a function that provides the spatially binned features.
A plot of the feature vector for a test image is presented in the following image:

![alt text][image4]

The next step was to explore the dataset provided (cars and non-cars), which will later be used to train the classifier.
The cars dataset consisted of 8792 images and that of the non-cars contained 8968 images.
Therefore, we can consider the combined dataset as balanced.  
The shape of each dataset entry was (64, 64, 3)  and the data type float32.

Image5 shows a car, and a non-car example:

![alt text][image5]

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.


##Histogram of Gradients (HOG) and other extracted features

Initially I calculated the HOG features for a random car, and non-car image, and visualised them.
HOG features were calculated after the original images were converted to grayscale.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Below is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.
Images 6 and 7 present an example of a random car image and the visualisation of its HOG features, as well as that of a random non-car, where the original image is displayed alongside its HOG visualisation:

![alt text][image6]

![alt text][image7]

In order to get the HOG features, I use the hog function, imported from skimage.feature.
I created a function that uses it, and passes the parameters the hog function requires.
Depending on whether a visualisation is desired or not, the function (called "get\_hog\_feature", which is defined at the 13th cell of the jupyter notebook) returns either only the features, or the visualisation in addition.

Then, I combined the spatial binning of colour, with the histograms of color, and the HOG features, creating a function that returns the combined vector of features.
What features are included, depends on the values of the parameters, provided to the function at runtime.
The function is the "extract\_features" function, which is defined in cell No. 15


####2. Explain how you settled on your final choice of HOG parameters.

In general, tried different parameter combinations, in order to decide which ones would be more suitable.
I used the YCrCb colorspace, as it seems to yield better results.

I generate the HOG for each color channel with 9 orientations, 8 pixels per cell, and 2 cells per block. 
This generates a feature vector of length 8460.

I tried various combinations, however chose them, because they seemed to have the highest prediction accuracy, when used by an SVM classifier in the next step.
In the jupyter notebook there is also an example with another set of parameters, that was suggested in the course forums it has the optimal results.
11 orientations, 16 pixels per cell, and 2 cells per block.
this yields a feature vector of length 2052.

I chose 9 orientations, 8 pixels per cell, and 2 cells per block, as it seemed to have a slightly higher test accuracy (0.9932, compared to the 0.9916 of the 11 orientations configuration).
The configuration with the 11 orientations was significantly faster to train though.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 20% of the dataset for testing (after randomising the dataset first).
The configuration used for all consequent steps in the assignment was the following:

colorspace: 'YCrCb'
orientations: 9
pixels per cell: 8
cells per block: 2
hog channels: ALL
Spatial binning size: (32, 32)
Number of histogram bins: 32
Min and max in y to search in slide\_window(): [450, None] (meaning max y is the maximum y of the image) 

HOG features are on
Histogram features are on
Spatial features are on

I trained the classifier, and stored it (together with the parameters) in a pickle file, so I could easily use them again.

In addition, I plotted the features of a car, and a non-car image:

![alt text][image8]


###Sliding Window Search

####1. Initially I used the slide\_window function used at the lectures to get a list of the windows that should be checked in an image, defining the area that will be considered, the size of each window, and the overlap factor.
In the next image these windows are drawn on one of the test images:

![alt text][image16]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Eventually, I used the "find\_cars" function (defined in cell No. 21), in order to  extract features using hog sub-sampling, and make predictions.
The parameters provided to this function are the image, the starting and the end point in the y dimension we will consider (cars will not usually be on the top part of the image), the scale we will use, the trained classifier, the scaler we use, and the parameters defining the features, such as orientations, pixels per cell, cells per block, spatial binning size, and number of histogram bins.
The function returns the image with the bounding boxes drawn, as well as a list of these bounding boxes.

An example using a test image, is displayed bellow:

![alt text][image9]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  

In general, the boxes close enough to each other are merged using the heatmap.
A class called "Vehicle" was created (Cell No. 28), that holds the bounding boxes of the last 6 frames (similar to the Line Class of the previous assignment)
The heatmap is constructed using these last 6 frames bounding boxes, and is then thresholded, in order to filter out false positives.
The threshold is one third of the length of the recent bounding boxes variable held by the Vehicle object.
The above helps additionally to reduce the jittering or jumping around of the detected bounding boxes.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, their corresponding heatmaps, and the resulting bounding boxes, drawn on the frames:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

####3. Combine Vehicle detection with Advanced Lane Detection from Previous Assignment

The video output from the previous assignment that already has the lane that we drive on marked, was fed as input to the vehicle detection pipeline. 
It would be possible to process everything in one step and create a common pipeline, it was not performed here for simplicity reasons, since the video with the lane marked was already available. 
Otherwise, tin order to create a common pipeline, it would simply require the image returned from the lane finding pipeline to be the input to the vehicle detection pipeline, and the output to be the combined result.
Here's a [link to my combined video result](./combined_project_video_output.mp4)

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The pipeline runs rather fast (about 6.8it/s), and on a better laptop could probably run in near real time.
It does mostly a good job identifying vehicles, and - at least in the test video - there are no false positives, as they are filtered out by the heatmap threshold. 

In my opinion, the biggest problem with the pipeline used, is that most parameters are hardcoded.
Therefore, it would be rather easy to fail in a different input video, unless of course trial and error is used, to select new parameter values.
Using additional data, would probably help the model generalise better, and have either less false positives in new images, or more vehicles successfully detected.
Additionally, other methods such as neural networks/deep learning could potentially provide more accurate results.
It can be however, that they also require more time to train.

Finally, sometimes there is a slight jittering in the bounding boxes, which means that there are probably more things that can be done, in order to smooth out the bounding boxes detected.

