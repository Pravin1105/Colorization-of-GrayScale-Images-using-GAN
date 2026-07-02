import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import cv2
SIZE = 128

model = load_model('/Users/pravin/Documents/VSCode/Colorizartion_GAN/generator_model.keras')

def preprocess_image(image_path):
  """
  Loads an image from the given path and preprocesses it for prediction.

  Args:
    image_path: The path to the image file.

  Returns:
    A tuple containing:
      - image_for_model: A preprocessed image as a NumPy array (1, SIZE, SIZE, 1) for model input.
      - image_grayscale: The original 2D grayscale image (SIZE, SIZE) for display.
  """

  # Load the image
  image = cv2.imread(image_path)

  # Resize the image
  image = cv2.resize(image, (SIZE, SIZE))

  # Convert the image to grayscale
  image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Normalize the image
  image_normalized = image_grayscale / 255.0

  # Add batch and channel dimensions for the model
  image_for_model = np.expand_dims(image_normalized, axis=0)   # (1, SIZE, SIZE)
  image_for_model = np.expand_dims(image_for_model, axis=-1)  # (1, SIZE, SIZE, 1)

  return image_for_model, image_grayscale

# Preprocess the image
image_path = 'Colorizartion_GAN/gray.jpeg'
image_for_model_input, original_grayscale_for_display = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(image_for_model_input)

# Show the prediction
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_grayscale_for_display, cmap='gray')
plt.title('Input Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prediction[0])
plt.title('Generated Color Image')
plt.axis('off')
plt.show()
