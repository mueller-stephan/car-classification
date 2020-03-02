import os

import numpy as np
import pandas as pd
import tensorflow as tf

from car_classifier.pipeline import construct_ds
from car_classifier.modeling import TransferModel
from car_classifier.utils import show_batch, show_batch_top_n, show_batch_with_pred

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# Gobal settings
INPUT_DATA_DIR = '../data/raw/'
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
TARGET = 'make'
BASE = 'ResNet'

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Create a list of all possible outcomes
if TARGET == 'make':
    classes = list(set([file.split('_')[0] for file in files]))
if TARGET == 'model':
    classes = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

# Targets in list
classes_lower = [x.lower() for x in classes]

# Split paths into train, valid, and test
files_train, files_test = train_test_split(file_paths, test_size=0.25)
files_train, files_valid = train_test_split(files_train, test_size=0.25)

# Construct tf.data.Dataset from file paths
ds_train = construct_ds(input_files=files_train, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE,
                        augment=False)
# ds_valid = construct_ds(input_files=files_valid, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE)
# ds_test = construct_ds(input_files=files_test, batch_size=BATCH_SIZE, classes=classes_lower, input_size=INPUT_SHAPE)

# for image, label in ds_train.take(1):
#     plt.imshow(image[0])
#     plt.title(classes[np.argmax(label[0])])
#     plt.show()
#     break

from tensorflow.keras.applications import ResNet50V2
from keras.applications.imagenet_utils import decode_predictions

# Use Complete ResNet model
model = ResNet50V2(include_top=True, input_shape=INPUT_SHAPE, weights='imagenet')
# scale problem with input? Still doesn't make sense for evaluate...
pred = model.predict(ds_train.take(1))
pred_labels = decode_predictions(pred, top=3)

print(pred_labels)

# Show examples from one batch
# plot_size = (18, 18)

# show_batch(ds_train, classes, size=plot_size)

# show_batch(ds_train, classes, size=plot_size, title='Training data')
# show_batch(ds_valid, classes, size=plot_size, title='Validation data')
# show_batch(ds_test, classes, size=plot_size, title='Testing data')

# Read in model, predict and plot

# model = TransferModel(base=BASE, shape=INPUT_SHAPE, classes=classes, dropout=0.2)
#
# model.compile(loss="categorical_crossentropy",
#               optimizer=Adam(0.0001),
#               metrics=["categorical_accuracy"])
#
# model.model.load_weights('../checkpoints/cp.ckpt')

# # Init base model and compile
# model = TransferModel(base=BASE,
#                       shape=INPUT_SHAPE,
#                       classes=classes, dropout=0.2)
#
# model.compile(loss="categorical_crossentropy",
#               optimizer=Adam(0.0001),
#               metrics=["categorical_accuracy"])
#
# # Load weights
# model.model.load_weights("../checkpoints/cp.ckpt")

# print("Weights loaded...")

# model.evaluate(ds_valid)

# show_batch_top_n(model, ds_valid, classes, rescale=True, size=plot_size)
# show_batch_with_pred(model, ds_valid, classes, rescale=True, size=plot_size)

# # Train model using defined tf.data.Datasets
# model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=10)
#
# # Plot accuracy on training and validation data sets
# model.plot()
#
# # Evaluate performance on testing data
# model.evaluate(ds_test=ds_test)
#
#
# # ---------
# test_batch = ds_train.take(1)
#
# p = model.predict(test_batch)
#
# pred = [np.argmax(x) for x in p]
#
# for img, lab in test_batch.as_numpy_iterator():
#     actual = np.argmax(lab, axis=1)
#
# pd.DataFrame({'actual': actual, 'pred': pred})
