```python
# New libriaries
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize pixel values (from range 0-255 to 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D (784 pixels)
    Dense(128, activation='relu'), # Hidden layer with 128 neurons
    Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    Dense(10, activation='softmax') # Output layer with 10 neurons (one for each class)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)


# Save the model
model.save('mnist_digit_classifier.h5')
print("Model saved as mnist_digit_classifier.h5")

# Load the model
loaded_model = load_model('mnist_digit_classifier.h5')
print("Model loaded successfully.")



# Image preprocessing:
    # Open the image using PIL
    image = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)

    # Resize the image to 28x28 (same size as MNIST images)
    image = image.resize((28, 28))

    # Convert the PIL image to a NumPy array
    image_array = img_to_array(image)

    # Normalize pixel values (scale from 0-255 to 0-1)
    image_array = image_array / 255.0

    # Reshape the image array to add a batch dimension (1, 28, 28, 1)
    image_array = np.expand_dims(image_array, axis=0)


# Predict digit on image using loaded_model and log number predicted.

# Example of usage:    
    # > python digit_classifier.py test_digit.jpg
    # > The predicted digit is: 5

```
