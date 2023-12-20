This project aims to recognize hand gestures using computer vision techniques and machine learning. It consists of three main components:

Create Gesture Dataset:

Captures hand gestures through a webcam.
Segments the hand region from the background.
Allows users to create a dataset of hand gesture images by adjusting their hand in front of the camera.

Train Model:

Utilizes a Convolutional Neural Network (CNN) implemented using TensorFlow and Keras.
Trains the model on the hand gesture dataset to recognize different gestures.
Provides visualizations of training images and evaluates model performance.

Predict Gesture:

Loads the pre-trained hand gesture recognition model.
Captures real-time hand gestures through the webcam.
Maps recognized gestures to predefined actions, such as adjusting volume, brightness, or opening applications.

This project can be used for various applications, including gesture-based control systems and human-computer interaction.
