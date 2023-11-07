## Assignment 7

Install
install tensorflow[and-cuda]
Implementation of MnistKeras class in mnist_keras.py
Build a two layer net using the Dense layer and Sequential Model.
First layer: 100 nodes
load_data()
 Load MNIST data using keras datasets. Use mnist.load_data() from tensorflow.keras.datasets.
Reshape them to the size of 28*28.
Normalize values from 0 - 255 to 0 - 1.
Return in the form of (x_train, y_train), (x_test, y_test)
build_model()
Build a model using Sequential
Return the model after compiling with proper optimizer, loss, and metrics.
train(model, model_name)
Train a model using fit() and save it in a file given as an argument of this function
Return the trained model
load(model_name)
Load a saved model using load_model()
Return the loaded model
test(model_name)
Test a model using evaluate() with x_test and y_test
Print accuracy and loss.
predict(x)
Return a prediction given input x. Use predict() function
Train.py
Use mnist_keras.py
Train the network and save a model
The work flow is something like
(x_train, y_train), (_, _) = load_data()
model = build_model()
model = train(model, "model_your_first_lastname")
Test.py
Use mnist_keras.py
Test the trained network
The work flow is something like
(_, _), (x_test, y_test) = load_data()
model = load("model_your_first_lastname")
test(model)
module7.ipynb
Show your train.py and test.py are properly working.
Test the trained model with your own hand-written images that you used in the previous assignments.
Find 12 unsuccessful predictions in the test set in random order 
Use shuffle of sklearn.utils to make the test set ‘shuffle’ randomly selected.
You may need to install sklearn in your conda environment.
Show them in 3 x 4 subplots.
Repeat twice to show the 12 unsuccessful predictions are from randomly selected samples.
Deliverables
Your GitHub repo must have the following files at your ece5831-2023-assignments/assignment-7/

mnist_keras.py
train.py
test.py 
module7.ipynb 