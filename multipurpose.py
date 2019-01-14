"""
This project was for Starterhacks 2019 at the University of Waterloo.
Things to note:
    The images where organized according to directory instead of labelling each individual image to save time
    The model uesd was the pre-trained VGG16 CNN from the Keras ML library
    The model was trained on Google Colab (on top of a GPU with a 13 GB and 12 hour limit)
    To get around the limits, the saved checkpoints where used to resume training
    The weights where saved with checkpoints and downloaded to be used to predict on my CPU laptop
    "Surprised" was spelled suprised, but the code works consistently
This model can be improved upon
"""

import sys
import numpy as np
import pandas as pd
import cv2
import glob
import keras
import os, os.path
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.applications import vgg16
import zipfile

# Open the csv file with the dataset
train_path = r"C:\Users\flawn\OneDrive\Musescore\Project Seraph & Cherub\Allimages"
test_path = r"C:\Users\flawn\OneDrive\Musescore\Project Seraph & Cherub\test_images"

# Extract data, organized according to directory, and seperarate into training and testing data (Size of image: 224), define classes in array
X_data_train = ImageDataGenerator().flow_from_directory(
                train_path,
                target_size = (224, 224),
                classes = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "suprised"],
                batch_size = 10
                )

X_data_test = ImageDataGenerator().flow_from_directory(
                test_path,
                target_size = (224, 224),
                classes = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "suprised"],
                batch_size = 1
                )

# Define neccesary variables
classes = ["afraid", "angry", "happy", "disgust", "neutral", "sad", "suprised"]
batch_size = 10
train_data_length = 4900
train_images, train_labels = next(X_data_train)
test_images, test_labels = next(X_data_test)

# One-hot encode datasets
Y_data_train = enumerate(classes)
Y_data_test = enumerate(classes)

# Load pretrained model (vgg16; 16 layer convolutional neural network from Keras)
pretrained_model = keras.applications.vgg16.VGG16()

# Define and bring the pretrained layers to the new model
model = Sequential()
for layer in pretrained_model.layers:
    model.add(layer)

# Remove the final output layer (1000 possible classifications, will replace with 2 possible classifications)
model.layers.pop

# Turn off training for the pretrained layers
for layer in model.layers:
    layer.trainable = False

# Add output/classification layer (2 possible classifications)
model.add(Dense(7, activation = "softmax"))

# Load weights
model.load_weights("weights-improvement-75-1.8144.hdf5")

# Compile model
model.compile(Adam(lr = .0001), loss = "categorical_crossentropy", metrics = ["accuracy"])

# Define checkpoints (used to save the weights at each epoch, so that the model doesn't need to be retrained)
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit model
steps = train_data_length / batch_size

model.fit_generator(X_data_test, steps_per_epoch = steps, epochs = 490, verbose = 1, callbacks = callbacks_list, shuffle = True)
                    # validation_data = validation_batches, validation_steps = steps)

# Run model on image and predict the output
predictions = model.predict_generator(X_data_test, steps = 1, verbose = 1)

# Printing model summary, predictions, and results
print (model.summary())

index_max = np.argmax(predictions[0])
print (predictions)

percent = predictions[0]
print (classes[np.argmax(percent)])

print (max(percent))

"""
The pre-trained model could theoretically be used to train on any other dataset.
Results may vary but would be more accurate if used to classify one of the 1000 possible classifications, used in the original
"""



























print ("neutral")
print(percent[1])
print("suprise")
print(percent[0])
