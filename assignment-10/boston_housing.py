import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import boston_housing   

class Boston_housing:
    def __init__(self, num_epochs=200, batch_size=1, validation_split=0.2, patience=3):
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.VALIDATION_SPLIT = validation_split
        self.PATIENCE = patience
        self.history = None
        self.model = None

    def load_data(self):
        (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
        return (train_data, train_labels), (test_data, test_labels)

    def normalize_data(self,data):
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        return data

    def build_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model    

    def fit(self, x_train, y_train):
        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = self.model.fit(x_train, y_train, epochs=self.NUM_EPOCHS,
                                batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT,
                                )
        self.history = history.history

    def plot_training_history(self):
        history_dict = self.history 
        history_dict.keys()
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'r-', label='training loss')
        plt.plot(epochs, val_loss_values, 'b--', label='validation loss')
        plt.title('training vs. validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()

        plt.show()

    def save_model(self, filename='model_boston_housing'):
        self.model.save(filename)

    def predict(self, x_data):
        return self.model.predict(x_data)