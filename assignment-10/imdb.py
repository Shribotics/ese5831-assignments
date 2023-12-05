import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import imdb

class Imdb:
    def __init__(self, num_words=10000, num_epochs=30, batch_size=512, validation_split=0.2, patience=3):
        self.NUM_WORDS = num_words
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.VALIDATION_SPLIT = validation_split
        self.PATIENCE = patience
        self.history = None
        self.model = None

    def load_data(self):
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=self.NUM_WORDS)
        return (train_data, train_labels), (test_data, test_labels)

    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.NUM_WORDS))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results 

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(self.NUM_WORDS,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def fit(self, x_train, y_train):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = self.model.fit(x_train, y_train, epochs=self.NUM_EPOCHS,
                                batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT,
                                callbacks=[callback])
        self.history = history.history

    def plot_training_history(self):
        loss_values = self.history['loss']
        val_loss_values = self.history['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'r-', label='training loss')
        plt.plot(epochs, val_loss_values, 'b--', label='validation loss')
        plt.title('Training vs. Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model(self, filename='model_imdb'):
        self.model.save(filename)