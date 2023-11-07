from mnist_keras import MnistKeras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import sys


if __name__ == "__main__":
    mnist_keras = MnistKeras()

    # Load the test data
    (_, _), (x_test, y_test) = mnist_keras.load_data()

    # Load the trained model
    model = mnist_keras.load("shrikant_mnist_nn_model.keras")

    # Test the model
    mnist_keras.test(model)

   # def load_images(file_path):
        ##images = images.resize((28, 28))
        #grey_img = images.convert("L")
        #img_array = np.array(grey_img)
        #img_array = img_array.reshape(1, 28*28)
        #return img_array/255.0
    
    #img_path = 'images/4_1.png'
    #img = load_images(img_path)
    #prediction = mnist_keras.predict(img)

    #predicated_digit = prediction.argmax()

    #p_hat = np.argmax(prediction)
    #p_certainty = prediction[0][predicated_digit]

