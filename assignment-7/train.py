from mnist_keras import MnistKeras

model_file_name="shrikant_mnist_nn_model.keras"

mnist_keras = MnistKeras()
(x_train, y_train), (_, _) = mnist_keras.load_data()

model = mnist_keras.build_model()
model = mnist_keras.train(model, model_file_name)