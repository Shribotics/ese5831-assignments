import gzip
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Mnist():
    def __init__(self):
        self.img_size = 28*28
        self.model_file_name = 'mnist/sample_weight.pkl'
        self.key_file = {
            'test_img':     'mnist/t10k-images-idx3-ubyte.gz',
            'test_label':   'mnist/t10k-labels-idx1-ubyte.gz'
        }
        self.network = self.init_network()

    #load images
    def load_images(self,file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.img_size)

        print('Done with loading images:', file_name)

        return images

    #load labels
    def load_labels(self,file_name):
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        print('Done with loading labels: ', file_name)
        return labels

    #define sigmoid function
    def sigmoid(self, a):
        return 1/(1 + np.exp(-a))

    #define softmax 
    def softmax(self,a):
        c = np.max(a)
        a = np.exp(a - c)
        s = np.sum(a)
        return a/s 
        
    #define predict function
    def predict(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3

        y =  self.softmax(a3)

        return y
    
    def init_network(self):
        with open(self.model_file_name, 'rb') as f:
            network = pickle.load(f)
        return network


    def main(self):
        x_test = self.load_images(self.key_file['test_img'])
        test_img = x_test[0000].reshape(28, 28)
        y_test = self.load_labels(self.key_file['test_label'])
        input_5000 = x_test[0000] / 255.0

        y = self.predict(input_5000)
        y_hat = np.argmax(y)
        y_certainty = y[y_hat]

        print(y)
        print(np.sum(y))
        if y_hat == y_test[0000]:
            print('Success')
        else:
            print('Fail')
        
        print(f'x[5000] is predicted as {y_hat} with {y_certainty*100}%. The label is {y_test[0000]}')


if __name__ == '__main__':
    mnist = Mnist()
    mnist.main()
    print("""/How to use the class /*'
              Step 1: Import numpy, pickel and gzip and create an instance of the class
              Step 2: Initaialize te constructors and then load labels and images
              Step 3: Use sigmoid and softmax functions
              Step 4: Predict function used a neural network
              Step 5: Initialize nneural network with pickle
              Step 6: Main function for testing purpose
        
        """)