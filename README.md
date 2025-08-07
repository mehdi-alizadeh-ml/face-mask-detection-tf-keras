# Face Mask Detection Using CNN

## Overview
This project detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model was trained on a labeled dataset of face images with and without masks.

## Dataset
- The dataset contains images of two classes: `with_mask` and `without_mask`.
- The data is split into training and validation sets.
- All input images are resized to 128x128 pixels.
- Class weighting was applied during training to handle class imbalance.


## Model Architecture
- A simple CNN model built from scratch using TensorFlow and Keras.
- Input image size: 128x128 pixels.
- Trained for 25 epochs with class weighting.

## Results
- Validation accuracy reached approximately 97%.
- Precision, recall, and F1-score above 0.98 for both classes.
- Confusion matrix and classification reports are included in the repository.

## Usage

### Requirements
- Python 3.x
- TensorFlow
- NumPy
- scikit-learn
- matplotlib

### Running the Model

1. Clone this repository:

Install the required libraries:

pip install tensorflow numpy scikit-learn matplotlib



Load the trained model and run predictions on new images:

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('mask_model.h5')
img = image.load_img('test_image.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
classes = ['with_mask', 'without_mask']
print(f"Prediction: {classes[np.argmax(prediction)]}")

   ```bash
   git clone <repository-url>
