import os
import shutil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory

class DogsCats:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        input = keras.Input(shape=(200, 180, 3))
        x = layers.Rescaling(1./255)(input)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=input, outputs=outputs)
        return model

    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def make_datasets(self, train_path, validation_path, test_path, batch_size=32, img_size=(200, 180)):
        # Function to create dataset directories
        def make_dataset(subset_name, start_idx, end_idx):
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
            labels='inferred',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
            seed=42,
            validation_split=0.2,
            subset='training'
        )
        return dataset

    def fit(self, train_dataset, validation_dataset, epochs=10, callbacks=None):
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def predict(self, model_name, file_name):
        # Load the trained model
        loaded_model = keras.models.load_model(model_name)

        # Preprocess the image
        img = image.load_img(file_name, target_size=(200, 180))
        img_array = image.img_to_array(img)
        img_array = img_array.reshape((3,) + img_array.shape)
        img_array /= 255.0  # Normalize the image

        # Make predictions
        predictions = loaded_model.predict(img_array)
        return predictions

# Set up paths
data_from_kaggle = "data-from-kaggle/train/train"
data_dirname = "dogs-vs-cats"
train_path = os.path.join(data_dirname, "train")
validation_path = os.path.join(data_dirname, "validation")
test_path = os.path.join(data_dirname, "test")

# Create DogsCats instance
dogs_cats = DogsCats()

# Make datasets
train_dataset, validation_dataset, test_dataset = dogs_cats.make_datasets(train_path, validation_path, test_path)

# Compile the model
dogs_cats.compile()

# Fit the model
callbacks = [keras.callbacks.ModelCheckpoint(
    filepath="model-from-scratch",
    save_best_only=False,
    monitor="val_loss"
)]
history = dogs_cats.fit(train_dataset, validation_dataset, epochs=10, callbacks=callbacks)

# Save the model
model_name = 'dogs_cats_model.pb'
dogs_cats.model.save(model_name)

# Example prediction
file_name = 'dog-vs-cat/test/dog.12491.jpg'
predictions = dogs_cats.predict(model_name, file_name)
print(predictions)
