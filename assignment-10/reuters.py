from keras.datasets import reuters  
import numpy as np
from keras import models
from keras.models import load_model
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf




class Reuters():
    
    def __init__(self):
        self.NUM_WORDS = 1000
        self.NUM_EPOCHS = 50
        self.BATCH_SIZE = 128
        self.VALIDATION_SPLIT = 0.2
        self.PATIENCE = 3
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = reuters.load_data(num_words=self.NUM_WORDS)


    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.NUM_WORDS))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results


    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(self.NUM_WORDS,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))  # 46 is the number of classes in the Reuters dataset

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = model.fit(self.vectorize_sequences(self.train_data), self.train_labels, 
                            epochs=self.NUM_EPOCHS, batch_size=self.BATCH_SIZE, 
                            validation_split=self.VALIDATION_SPLIT, callbacks=[callback])
        
        model.save('Reuters_model')

        print("Plotting Training and Validation Loss")

        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'r-', label='training loss')
        plt.plot(epochs, val_loss_values, 'b--', label='validation loss')
        plt.show()


        # return history

    def predict(self, model):
        print(model.predict(self.vectorize_sequences(self.test_data)))






