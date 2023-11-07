import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras.models import load_model



class MnistKeras:
    def __init__(self):
        pass
    
    def load_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_data) = mnist.load_data()
        input_size = self.train_images.shape[1]* self.train_images.shape[2]
        train_size = self.train_images.shape[0]
        test_size = self.test_images.shape[0]
        self.train_images = self.train_images.reshape(train_size, input_size)
        self.train_images = self.train_images.astype("float32") /255
        self.test_images = self.test_images.reshape(test_size, input_size)
        self.test_images = self.test_images.astype("float32") /255

        return (self.train_images, self.train_labels), (self.test_images, self.test_data)
    
    def build_model(self):
        model = keras.Sequential([
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer= "rmsprop" ,
              loss= "sparse_categorical_crossentropy" ,
              metrics= "accuracy")
        
        return model
    
    def train(self, model, model_file_name):
        model.fit(self.train_images, self.train_labels, epochs=30, batch_size=64)
        model.save(model_file_name)

        return model
    
    def load(self, model_file_name):
        self.loaded_model = load_model(model_file_name)
        
        return self.loaded_model
    
    def test(self, model):
        metrics = model.evaluate(self.test_images, self.test_data)
        print(f"Accuracy: {metrics[1]*100}% \t Loss: {metrics[0]}")

    def predict(self, x):
        predictions = self.predict(x)

        return predictions