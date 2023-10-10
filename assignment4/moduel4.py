import numpy as np
from multilayer_perceptron import MultilyerPerceptron

#case 1:
mlp = MultilyerPerceptron()
y = mlp.forward(np.array([[0.25423, 0.1694464]]))
print(y)

#case 1:
mlp = MultilyerPerceptron()
y = mlp.sigmoid(np.array([[0.468454, 0.1452]]))
print(y)


#case 1:
mlp = MultilyerPerceptron()
y = mlp.identity_function(np.array([[0.749, 0.484445]]))
print(y)

