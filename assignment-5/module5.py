import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import sys


class Mnist:
    def __init__(self, img_path, digit):
        self.img_size = 28 * 28
        self.model_file_name = 'mnist/sample_weight.pkl'
        self.image_path = img_path  # Define the image path here
        self.digit = digit  # Define the label digit here
        self.network = self.init_network()

    def load_images(self, file_path):
        images = Image.open(file_path)
        #images.show()
        images = images.resize((28, 28))
        grey_img = images.convert("L")
        img_array = np.array(grey_img)
        img_array = img_array.reshape(28*28)

        return img_array / 255.0

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def softmax(self, a):
        c = np.max(a)
        a = np.exp(a - c)
        s = np.sum(a)
        return a / s

    def predict(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)
        return y

    def init_network(self):
        with open(self.model_file_name, 'rb') as f:
            network = pickle.load(f)
        return network

    def main(self):
        input_image = self.load_images(self.image_path)

        y = self.predict(input_image)
        y_hat = np.argmax(y)
        y_certainty = y[y_hat]

        print(y)
        print(np.sum(y))
        if y_hat== self.digit:
            print(f'Success: Image at the location {self.image_path} is predicted as {y_hat} with {y_certainty * 100}%. The expected label is {self.digit}')
        else:
            print(f'Fail: Image at the location {self.image_path} is predicted as {y_hat} with {y_certainty * 100}%. The expected label is {self.digit}')

        #print(f'Image at the location {self.image_path} is predicted as {y_hat} with {y_certainty * 100}%. The expected label is {self.digit}')



if __name__ == "__main__":
    path = str(sys.argv[1])
    num = int(sys.argv[2])
    mnist = Mnist(path,num)
    mnist.main()
