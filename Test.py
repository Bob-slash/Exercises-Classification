import os
import cv2
import numpy as np
import tensorflow as tf


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="workout_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
data = []
labels = []
DATA_DIR = "archive/"
CLASSES = os.listdir(DATA_DIR)

#Test Data
image = cv2.imread("bench press_Test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
data.append(image)
data = np.array(data)
labels.append(0)



# Test the model on input data.
input_shape = np.shape(data)
input_data = np.array(data, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

#Prints probabilities of each excersize
output_data = interpreter.get_tensor(output_details[0]['index'])

max_index = 0
max_val = 0
for i in range(len(output_data[0])):
    if output_data[0][i] > max_val:
        max_val = output_data[0][i]
        max_index = i

DATA_DIR = "archive/"
CLASSES = os.listdir(DATA_DIR)
print("Prediction: " + CLASSES[max_index])
