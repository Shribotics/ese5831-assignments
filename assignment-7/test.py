from mnist_keras import MnistKeras

if __name__ == "__main__":
    mnist_keras = MnistKeras()

    # Load the test data
    (_, _), (x_test, y_test) = mnist_keras.load_data()

    # Load the trained model
    model = mnist_keras.load("model_your_first_lastname.h5")

    # Test the model
    mnist_keras.test(model)