#**Traffic Sign Recognition** 

Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./writeup_images/test_images.png "Test Images"
[image3]: ./writeup_images/train_modle.png "Train Modle"
[image4]: ./writeup_images/children_crossing.jpeg "New Image 1"
[image5]: ./writeup_images/no_passing.jpeg "New Image 2"
[image6]: ./writeup_images/road_work.jpeg "New Image 3"
[image7]: ./writeup_images/slippery_road.jpeg "New Image 4"
[image8]: ./writeup_images/stop.jpeg "New Image 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Steps to preprocess image data:
1, grayscale.
2, resize if necessary.
3, equalize.
4, normalize.

As a first step, I decided to convert the images to grayscale because it can prevent color from adding additional hurdles to training the data.

Then I resized the images as data need to be in the same size for train for later computation.

Then I equalized the image to clean some low colors.

Finally I normalized images to improve the contrast of the images.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        			| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				| 
| Convolution layer 1   | 5x5 windows, 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					| activate 1					|
| Max pooling	      	| 2x2 stride, outputs 14x14x16 					|
| Convolution layer 2   | 5x5 windows, 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					| activate 2					|
| Max pooling	      	| 2x2 stride, outputs 5x5x32 					|
| Fully connected layer 1| inputs 800, outputs 512 					|
| RELU				| activate 3     					|
| Fully Connected Layer 2| inputs 512, outputs 128					|
| RELU					 | activate4					|
| Fully Connected Final Layer	| inputs 128, outputs n_classes				|


3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Here is my code to train the model.

![alt text][image3]

I ran this modle on my Macbook Pro with i7. It took about 10 minutes to output the model. I set batch size to 256 and the accuracy looked good.

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 94.7%
* test set accuracy of 93.5%

I looked into a lot of solutions and finally came up with this one. I am still investigating those solutions.

Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image is most difficult to classify. I don't know why. My model recognized it as "road work" with only 47%.

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| children_crossing     | children_crossing   									| 
| no_passing     		| no_passing 										|
| road_work				| road_work											|
| slippery_road	      	| slippery_road					 				|
| stop			| stop      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. I think probably the number of my samples are quite small, so the accuracy might be high. If I continue to test the model with new images, I believe the accuracy will go down.

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last second cell of the Ipython notebook.

The followings are top 5 softmax probabilities for each image.

**Top 5 class predictions for children_crossing.jpeg:**

Correct ID: 28, label: Children crossing
Predicted ID 1: 28, label: Children crossing, probability: 0.8041014075
Predicted ID 2: 23, label: Slippery road, probability: 0.0823512897
Predicted ID 3: 11, label: Right-of-way at the next intersection, probability: 0.0344926640
Predicted ID 4: 38, label: Keep right, probability: 0.0302122943
Predicted ID 5: 20, label: Dangerous curve to the right, probability: 0.0188851152

**Top 5 class predictions for no_passing.jpeg:**

Correct ID: 9, label: No passing

Predicted ID 1: 17, label: No entry, probability: 0.7137765884

Predicted ID 2: 14, label: Stop, probability: 0.2848145664

Predicted ID 3: 9, label: No passing, probability: 0.0011010527

Predicted ID 4: 8, label: Speed limit (120km/h), probability: 0.0002295068

Predicted ID 5: 36, label: Go straight or right, probability: 0.0000692831
	
**Top 5 class predictions for road_work.jpeg:**

Correct ID: 25, label: Road work

Predicted ID 1: 25, label: Road work, probability: 0.4727835059

Predicted ID 2: 21, label: Double curve, probability: 0.4030152261

Predicted ID 3: 11, label: Right-of-way at the next intersection, probability: 0.0877435356

Predicted ID 4: 35, label: Ahead only, probability: 0.0216240752

Predicted ID 5: 26, label: Traffic signals, probability: 0.0103566507
	
**Top 5 class predictions for slippery_road.jpeg:**

Correct ID: 23, label: Slippery road

Predicted ID 1: 23, label: Slippery road, probability: 0.9962728024

Predicted ID 2: 19, label: Dangerous curve to the left, probability: 0.0037270512

Predicted ID 3: 21, label: Double curve, probability: 0.0000001036

Predicted ID 4: 11, label: Right-of-way at the next intersection, probability: 0.0000000747

Predicted ID 5: 31, label: Wild animals crossing, probability: 0.0000000035
	
**Top 5 class predictions for stop.jpeg:**

Correct ID: 14, label: Stop

Predicted ID 1: 14, label: Stop, probability: 0.9999716282

Predicted ID 2: 17, label: No entry, probability: 0.0000275515

Predicted ID 3: 38, label: Keep right, probability: 0.0000006575

Predicted ID 4: 34, label: Turn left ahead, probability: 0.0000000992

Predicted ID 5: 3, label: Speed limit (60km/h), probability: 0.0000000061

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


