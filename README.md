# **Traffic Sign Recognition**


**Still underdevelopment**
---

**Build a Traffic Sign Recognition Project with TensorFlow 2.0**

More examples can be found at [TensorFlow tutorials](https://www.tensorflow.org/tutorials/images/cnn)

Raw data from Kaggle: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
Each image in raw downloaded data has different size, thus they were processed to have a consistent shape. After processes including: resizing, splitting into training and validation, conversion to grayscale and normalization.
* Number of training examples = 31367
* Number of validation examples = 7842
* Number of testing examples = 12630
* Image data shape = (32, 32, 1)
* Number of classes = 43


My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image  			    		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		     		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 		     		|
| Fully connected		| Input = 400. Output = 120.        			|
| Fully connected		| Input = 120. Output = 84.        			|
| Dropout	          	|         			|
| Fully connected		| Input = 84. Output = 43.        			|
| Softmax				| Output layer      							|


My final model results were:
* validation set accuracy of ~ 0.96
* test set accuracy of ~ 0.92
