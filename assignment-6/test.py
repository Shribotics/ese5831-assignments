import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import sys
import two_layer_net as tl


def load_images(file_path):
        images = Image.open(file_path)
        #images.show()
        images = images.resize((28, 28))
        grey_img = images.convert("L")
        img_array = np.array(grey_img)
        img_array = img_array.reshape(28*28)
        return img_array/255.0


img_size = 28*28
model_path = 'shrikant_mnist_nn_model.pkl'
img_path = 'images/7_1.png'

with open(model_path, 'rb') as f:
    network = pickle.load(f)

network_object = tl.TwoLayerNet

y = network_object.predict(network, load_images(img_path))

y_hat = np.argmax(y)
y_certainty = y[y_hat]

print(y_hat)
print(np.sum(y))
print(y_certainty)

print(f'Image at the location is predicted as {y_hat} with {y_certainty * 100}%.')


