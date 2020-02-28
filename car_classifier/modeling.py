import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class TransferModel:

    def __init__(self, base: str, shape: tuple, classes: list, freeze: list = None, dropout: float = 0,
                 checkpoint_dir: str = "./checkpoints"):
        """
        Class for transfer learning from either VGG16 or ResNet

        Args:
            base: String giving the name of the base model (either 'VGG16' or 'ResNet')
            shape: Input shape as tuple (height, width, channels)
            classes: List of class labels
        """
        self.shape = shape
        self.classes = classes
        self.checkpoint_dir = checkpoint_dir
        self.history = None
        self.base = None
        self.model = None
        self.freeze = None

        # Class allows for two base models (VGG16 oder ResNet)
        # Use pre-trained ResNet model
        if base == 'ResNet':
            self.base_model = ResNet50V2(include_top=False,
                                         input_shape=self.shape,
                                         weights='imagenet')

            self.base_model.trainable = True
            if freeze is not None:
                self.base_model = self._freeze(model=self.base_model, patterns=freeze)

            add_to_base = self.base_model.output
            add_to_base = GlobalAveragePooling2D(data_format='channels_last', name='head_gap')(add_to_base)

        # Use pre-trained VGG16
        elif base == 'VGG16':
            self.base_model = VGG16(include_top=False,
                                    input_shape=self.shape,
                                    weights='imagenet')

            # Set entire model to be trainable
            # > 23e6 trainable parameters
            self.base_model.trainable = True

            if freeze is not None:
                self.base_model = self._freeze(model=self.base_model, patterns=freeze)

            add_to_base = self.base_model.output
            add_to_base = Flatten(name='head_flatten')(add_to_base)
            add_to_base = Dense(1024, activation='relu', name='head_fc_1')(add_to_base)
            add_to_base = Dropout(dropout, name='head_drop_1')(add_to_base)
            add_to_base = Dense(1024, activation='relu', name='head_fc_2')(add_to_base)
            add_to_base = Dropout(dropout, name='head_drop_2')(add_to_base)

        # Add final output layer
        add_to_base = Dense(len(self.classes), activation='softmax', name='head_pred')(add_to_base)
        new_output = Dropout(dropout, name='head_pred_dropout')(add_to_base)
        self.model = Model(self.base_model.input, new_output)

        # Model overview
        layers = [(layer, layer.name, layer.trainable) for layer in self.model.layers]
        self.freeze = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

    @staticmethod
    def _freeze(model, patterns: list):
        """
        Helper function to make certain layers trainable

        Args:
            model: tf.Model
            patterns: list of patterns as str to match layer names

        Returns:
            model
        """
        for layer in model.layers:
            for pattern in patterns:
                regex = re.compile(pattern)
                if regex.search(layer.name):
                    layer.trainable = False
                else:
                    pass

        return model

    def compile(self, **kwargs):
        """
        Compile method
        """

        self.model.compile(**kwargs)

    def train(self,
              ds_train: tf.data.Dataset,
              epochs: int,
              ds_valid: tf.data.Dataset = None):
        """
        Training method

        Args:
            ds_train: training data as tf.data.Dataset
            ds_valid: validation data as tf.data.Dataset
            epochs: number of epochs to train

        Returns
            Training history in self.history
        """

        # Define early stopping as callback
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       restore_best_weights=True)

        checkpoint_path = self.checkpoint_dir + '/cp.ckpt'

        checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

        callbacks = [early_stopping, checkpoint]

        # Fitting
        self.history = self.model.fit(ds_train,
                                      epochs=epochs,
                                      validation_data=ds_valid,
                                      callbacks=callbacks)

        return self.history

    def evaluate(self, ds_test: tf.data.Dataset):
        """
        Evaluation method

        Args:
            ds_test: Testing data as tf.data.Dataset

        Returns:
              None
        """

        # TODO add a function return
        self.model.evaluate(ds_test)

    def predict(self, ds_new: tf.data.Dataset, proba: bool = True):
        """
        Prediction method

        Args:
            ds_new: New data as tf.data.Dataset
            proba: Boolean if probabilities should be returned

        Returns:
            class labels or probabilities
        """

        p = self.model.predict(ds_new)

        if proba:
            return p
        else:
            return [np.argmax(x) for x in p]

    def top_n(self, ds_new: tf.data.Dataset, n=5):
        """
        Return probabilities of top n classes of prediction
        Args:
            ds_new: tf.data.Dataset to be predicted
            n: number of predictions to return

        Returns:
            list of dicts of length n with items {class label: probability}
        """

        p = self.model.predict(ds_new)

        assert(n <= len(p))

        class_probs = [sorted(list(enumerate(x)), key=lambda x: x[1])[-n:] for x in p]

        return class_probs

    def plot(self, what: str = 'metric'):
        """
        Method for training/validation visualization
        Takes self.history and plots it
        """

        if self.history is None:
            AttributeError("No training history available, call TransferModel.train first")

        if what not in ['metric', 'loss']:
            AttributeError(f'type must be either "loss" or "metric"')

        if what == 'metric':
            metric = self.model.metrics_names[1]
            y_1 = self.history.history[metric]
            y_2 = self.history.history['val_' + metric]
            y_label = metric
        elif what == 'loss':
            y_1 = self.history.history['loss']
            y_2 = self.history.history['val_loss']
            y_label = 'loss'

        plt.plot(y_1)
        plt.plot(y_2)
        plt.title('Model Performance')
        plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()



