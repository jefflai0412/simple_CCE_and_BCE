import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from PIL import Image
import wandb
from wandb.integration.keras import WandbCallback

class CategoricalCrossEntropyModel:
    def __init__(self):
        self.network = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def categorical_cross_entropy(self, yHat, y):
        epsilon = K.epsilon()  # Small value for numerical stability
        yHat = K.clip(yHat, epsilon, 1.0 - epsilon)
        return -K.mean(K.sum(y * K.log(yHat), axis=-1))

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        self.train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
        self.test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
        self.train_labels = to_categorical(train_labels)
        self.test_labels = to_categorical(test_labels)

    def build_model(self, input_shape, num_classes):
        self.network = Sequential()
        self.network.add(Dense(512, activation='relu', input_shape=(input_shape,)))
        self.network.add(Dense(num_classes, activation='softmax'))
        self.network.compile(optimizer='rmsprop', loss=self.categorical_cross_entropy, metrics=['accuracy'])

    def train_and_evaluate(self, epochs=5, batch_size=400):
        # Initialize Weights & Biases
        wandb.init(project="mnist-training", name="cross-entropy-experiment", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "rmsprop"
        })

        self.network.fit(
            self.train_images,
            self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.test_images, self.test_labels),
            callbacks=[WandbCallback(save_graph=False, save_model=False)]  # Disable model saving
        )

        # Evaluate the model
        test_loss, test_acc = self.network.evaluate(self.test_images, self.test_labels)
        print("Categorical Test Accuracy:", test_acc)

class BinaryCrossEntropyModel:
    def __init__(self):
        self.network = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.digits = None

    def binary_cross_entropy(self, yHat, y):
        epsilon = K.epsilon()  # Small value for numerical stability
        yHat = K.clip(yHat, epsilon, 1.0 - epsilon)
        return -K.mean(y * K.log(yHat) + (1 - y) * K.log(1 - yHat))

    def load_data(self, digits):
        self.digits = digits
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_filter = np.where(np.isin(train_labels, digits))
        test_filter = np.where(np.isin(test_labels, digits))

        self.train_images = train_images[train_filter].reshape((-1, 28, 28, 1)).astype('float32') / 255
        self.test_images = test_images[test_filter].reshape((-1, 28, 28, 1)).astype('float32') / 255
        self.train_labels = np.where(train_labels[train_filter] == digits[0], 0, 1).astype('float32')
        self.test_labels = np.where(test_labels[test_filter] == digits[0], 0, 1).astype('float32')

    def build_model(self, input_shape):
        self.network = Sequential()
        self.network.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        self.network.add(MaxPooling2D((2, 2)))
        self.network.add(Flatten())
        self.network.add(Dense(512, activation='relu'))
        self.network.add(Dropout(0.5))
        self.network.add(Dense(1, activation='sigmoid'))

        self.network.compile(optimizer='rmsprop', loss=self.binary_cross_entropy, metrics=['accuracy'])

    def train_and_evaluate(self, epochs=5, batch_size=200):
        # Initialize Weights & Biases
        wandb.init(project="mnist-training", name="binary-cross-entropy-experiment", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "rmsprop",
            "digits": self.digits
        })

        self.network.fit(
            self.train_images,
            self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.test_images, self.test_labels),
            callbacks=[WandbCallback(save_graph=False, save_model=False)]  # Disable model saving
        )
        

        # Evaluate the model
        test_loss, test_acc = self.network.evaluate(self.test_images, self.test_labels)
        print("Binary Test Accuracy:", test_acc)

    def predict(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img = np.array(img).astype('float32') / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)  # Add batch dimension
    
        # Predict
        prediction = self.network.predict(img)
        print(f"Predicted probability: {prediction[0][0]:.4f}")
        print(f"Predicted class: {self.digits[1] if prediction[0][0] > 0.5 else self.digits[0]}")


# Train and evaluate categorical cross-entropy model
# cat_model = CategoricalCrossEntropyModel()
# cat_model.load_data()
# cat_model.build_model(input_shape=28 * 28, num_classes=10)
# cat_model.train_and_evaluate()

# Train and evaluate binary cross-entropy model
bin_model = BinaryCrossEntropyModel()
bin_model.load_data(digits=[0, 4])
bin_model.build_model(input_shape=(28, 28, 1))
bin_model.train_and_evaluate()

# Example prediction
bin_model.predict("img/4.png")
