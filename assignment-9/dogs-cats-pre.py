import os
import shutil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input



data_dirname = "dogs-vs-cats"

class DogsCatsPre:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 180, 3))
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def make_datasets(self, train_path, validation_path, test_path, batch_size=32, img_size=(200, 180)):
        def make_dataset(subset_name, start_idx, end_idx):
            data_from_kaggle = "data-from-kaggle/train/train"
            data_dirname = "dogs-vs-cats"
            for category in {"cat", "dog"}:
                subset_path = os.path.join(data_dirname, subset_name, category)
                os.makedirs(subset_path, exist_ok=True)

                for i in range(start_idx, end_idx):
                    filename = f"{category}.{i}.jpg"
                    src = os.path.join(data_from_kaggle, filename)
                    dst = os.path.join(subset_path, filename)
                    shutil.copyfile(src, dst)

        # Create datasets with the specified number of samples
        make_dataset("train", 0, 1000)
        make_dataset("validation", 1001, 1500)
        make_dataset("test", 1501, 2500)

        # Load datasets
        train_dataset = self.create_dataset(train_path, batch_size, img_size)
        validation_dataset = self.create_dataset(validation_path, batch_size, img_size)
        test_dataset = self.create_dataset(test_path, batch_size, img_size)

        return train_dataset, validation_dataset, test_dataset
      

    def create_dataset(self, directory, batch_size, img_size):
    
        dataset = image_dataset_from_directory(
            directory,
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True
        )
        return dataset.map(self.preprocess)

    def preprocess(self, image, label):
        image = preprocess_input(image)
        return image, label

    def fit(self, train_dataset, validation_dataset, epochs=10, callbacks=None):
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def predict(self, model_name, file_name):
        # The predict function remains the same as in the previous script

        # Load the trained model
        loaded_model = keras.models.load_model(model_name)

        # Preprocess the image
        img = image.load_img(file_name, target_size=(200, 180))
        img_array = image.img_to_array(img)
        img_array = img_array.reshape(1,200,180,3)
        # img_array /= 255.0  # Normalize the image
        # Make predictions
        predictions = loaded_model.predict(img_array)
        return predictions
    
# Set up paths


