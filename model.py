import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128

DATA_DIR = "archive/"
CLASSES = os.listdir(DATA_DIR)
print(CLASSES)

with open('workout_label.txt', 'w') as f:
    for workout_class in CLASSES:
        f.write(f'{workout_class}\n')

data = []
labels = []

for dirname, _, filenames in os.walk(DATA_DIR):
    data_class = dirname.split(os.path.sep)[-1]
    print(data_class)
    for filename in filenames:
        img_path = os.path.join(dirname, filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

        data.append(image)
        labels.append(CLASSES.index(data_class))

data = np.array(data)
labels = np.array(labels)


labels = to_categorical(labels)

features_train, features_test, labels_train, labels_test = train_test_split(data,labels,test_size = 0.1,shuffle = True,random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rescale = 1./255,
#     rotation_range = 30,
#     zoom_range = 0.15,
#     width_shift_range = 0.2,
#     height_shift_range = 0.2,
#     shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)

trainAug.fit(features_train)

# initialize the validation/testing data augmentation object
valAug = ImageDataGenerator(rescale = 1./255)

valAug.fit(features_test)


def create_model():
    model = tf.keras.models.Sequential()

    # Convolution layers
    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))

    #     model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
    #     model.add(tf.keras.layers.MaxPool2D((2,2)))

    #     model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
    #     model.add(tf.keras.layers.MaxPool2D((2,2)))

    #     model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
    #     model.add(tf.keras.layers.MaxPool2D((2,2)))

    #     model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
    #     model.add(tf.keras.layers.MaxPool2D((2,2)))

    # Hidden layers
    model.add(tf.keras.layers.Flatten())
    #     model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(len(CLASSES), activation='softmax'))

    model.summary()

    return model

# Construct the model
workout_model = create_model()

# Display the success message.
print("Model Created Successfully!")

EPOCH = 10

# Create an Instance of Early Stopping Callback.
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 5,
                                        mode = 'min',
                                        restore_best_weights = True)

# compile our model (this needs to be done after our setting our layers to being non-trainable)
opt = tf.keras.optimizers.Adam()

workout_model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['acc']
             )
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
model_history = workout_model.fit(trainAug.flow(features_train, y = labels_train),
                                  epochs = EPOCH,
                                  batch_size = 128,
                                  validation_data = valAug.flow(features_test, labels_test),
                                  shuffle = True,
                                  callbacks = [early_stopping_callback]
                                 )

# evaluate the network
model_evaluation_history = workout_model.evaluate(features_test, labels_test)

# plot the training loss and accuracy
epoch = range(len(model_history.history["loss"]))
plt.figure()
plt.plot(epoch, model_history.history['loss'], 'red', label = 'train_loss')
plt.plot(epoch, model_history.history['val_loss'], 'blue', label = 'val_loss')
plt.plot(epoch, model_history.history['acc'], 'orange', label = 'train_acc')
plt.plot(epoch, model_history.history['val_acc'], 'green', label = 'val_acc')
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# Save Model to .h5 format
workout_model.save('workout_model')

# Convert the .h5 model to tflite
converter = tf.lite.TFLiteConverter.from_saved_model('./workout_model')
tflite_model = converter.convert()

# Save the tflite model
with open('workout_model.tflite', 'wb') as f:
    f.write(tflite_model)
