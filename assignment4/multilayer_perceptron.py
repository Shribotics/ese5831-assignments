import numpy as np
import matplotlib.pyplot as plt

class MultilyerPerceptron():
    def __init__(self):
        self.network = {}
        self.network['w1'] = np.array([[0.5,0.6,0.7],[0.1,0.2,0.3]])
        self.network ['b1'] = np.array([0.2,0.3,0.9])
        self.network['w2'] = np.array([[0.5,0.6],[0.1,0.2],[0.3,0.7]])
        self.network ['b2'] = np.array([0.2,0.9])
        self.network['w3'] = np.array([[0.5,0.4],[0.1254,0.2463]])
        self.network ['b3'] = np.array([0.2546164,0.9565465])

    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))

    def identity_function(self, s):
        return s
    
    def forward(self, x):
        w1,w2,w3 = self.network['w1'], self.network['w2'], self.network['w3']
        b1,b2,b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x,w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1,w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2,w3) + b3
        y = self.identity_function(a3)
        return y
