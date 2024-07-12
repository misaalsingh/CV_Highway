# CV_Highway

Purpose:
- The purpose of this project is to both experiment with computer vision techniques and learn more about Deep Learning as a whole
- The dataset I have used is recorded tapes on a highway near Seattle taken from a camera 

Dataset:
- The dataset I have used is recorded tapes on a highway near Seattle taken from a camera
- The camera has taken about 253 videos on this highway and these videos have been labeled with traffic levels of light, medium or heavy
- info.txt gives us the labels for each of the different highway tapes

Step Zero:
- Use the OpenCV package in python to read and preprocess these videos and prepare them to be modeled
- Extract the labels for each of the videos in order to help train any models I use

Step One:
- Use the tensorflow Inceptionv3 model to extract key features from the videos: the output of this will be a (52, 2048) np array (frames, features)
- Train/Fit the NN on the images and labels
- Evaluate the accuracy of the model

Step Two: 
- This portion will be done with CV2 & Yolov9
- Use cv2 and the pretrained yolo model to identify cars on the highway
- From this I can use numpy and CV2 to asses the displacement of each object and calculate its speed to see which cars are speeding
