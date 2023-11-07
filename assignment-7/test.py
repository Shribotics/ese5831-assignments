from mnist_keras import MnistKeras

if __name__ == "__main__":
    mnist_keras = MnistKeras()

    # Load the test data
    (_, _), (x_test, y_test) = mnist_keras.load_data()

    # Load the trained model
    model = mnist_keras.load("shrikant_mnist_nn_model.keras")

    # Test the model
    mnist_keras.test(model)