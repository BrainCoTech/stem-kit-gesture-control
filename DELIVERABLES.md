# STEM kit hand gesture recognition deliberables
- [X] Project goal briefly
- [X] List all dependencies and provide detailed setup instructions
- [X] Source code (Python and Arduino, clean up, organize & comment well)
- [X] Briefly discuss all methods you tried that didn't work well and why
- [ ] What is the unfinished portion (Advise on finishing up the unfinished portion)
- [X] Potential issues (Advice on future research/development)
- [ ] Conclusion (optional)



## Project goal
Develop a game in which users can play Rock, Paper, Scissors game with a Prosthetic hand by using their computer webcam. 
This is a simple application of Convolution Neural Networks combined with background elimination to detect different hand gestures. A background elimination algorithm extracts the hand image from the webcam and uses it to train and predict which type of hand gesture it is.



## Dependencies
#### Python setup
* Python3
* Tensorflow
* TfLearn
* Opencv (cv2) for python3
* Numpy
* Pillow (PIL)
* Imutils

To set up the environment, type pip install -r requirements.txt in terminal


#### Arduino setup
* Arduino IDE
Sketch  ->  Include libraries  -> search IRremote (by shirriff) -> Install


## File Description

[CommandCode.ino]: Run this file in ArduinoC IDE and upload it on the board. 

[PalmTracker.py]: Run this file to generate custom datasets. 

[ResizeImages.py]: Run this file after PalmTracker.py in order to resize the images so that it can be fed into the Convolution Neural Network designed using tensorflow. The network accepts 89px x 100px images.

[ModelTrainer.py] : This is the model trainer file. Run this file if you want to retrain the model using your custom dataset.

[ContinuousGesturePredictor.py]: Running this file opens up your webcam, takes continuous frames of your hand image, and then predicts the class of your hand gesture in realtime.

## How to run the RealTime prediction

Run the ContinuousGesturePredictor.py file and you will see a window named Video Feed appear on screen. Wait for a while until a window named Thresholded appears.

The next step involves pressing "s" on your keyboard in order to start the real-time prediction.

Bring your hand in the Green Box drawn inside Video Feed window in order to see the predictions. 


## Methods that I tried but failed

This background elimination algorithm is highly sensitive to lighting conditions and backgrounds. Therefore, the algorithm that works in one environmental setting fails to work in another. In order to solve the problem, we've tried different methods. 

1. Skin detection method 
The main drawback is that we are framing skin detection as a “color detection” problem. This assumes that we can easily specify the HSV values for ranges of pixel intensities that are considered skin.
Since only a range of color is given to this color detection method, this method does not perform well enough under different lighting conditions. Another drawback of this method is that we have to assume users are all in the same ethnic group.
This method does a little bit better than the background elimination under different backgrounds, but still does not reach our goal.


2. Three channels method
Instead of converting the images to grayscale, we split the images into 3 color channels--red, green, blue. Green and blue channels perform the best under different lighting conditions, we trained the model for both of them separately. However, I don't see any significant improvement. 




3.  3D Hand Pose Annotations
The main drawback of this method is the high hardware requirement. 
I tested this method on my Macbook Pro ( i5, 16GB ) with CPU.  Unfortunately, the fps I got was around 4, which is impossible to use it for our real time gesture recognition need. I've also tried reducing the webcam resolution as much as possible, but got no visible improvements.
TensorFlow GPU support only works on Linux. Therefore, the method does not fit our goal.

    https://github.com/timctho/convolutional-pose-machines-tensorflow

    https://github.com/guiggh/hand_pose_action




## Advise & thoughts



1. transfer learning (Unfinished)

2. infrared camera
An infrared camera might perform better than visible-light camera in that it has a much narrower band of light that it's sensitive to.  Its narrow light sensitivity spectrum not only could filter out unwanted noises and variations introduced by different lighting conditions but also highlight our object of interest, the human hand, because it gives off more heat than other objects in the environment. 
