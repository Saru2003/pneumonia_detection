import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('pneumonia_detection_saved_model/1')

def lambda_handler(event, context):
    # Parse the input data
    input_data = json.loads(event['body'])
    images = input_data['images']  # Assume images are provided in a list

    # Preprocess the images
    processed_images = preprocess_images(images)

    # Make predictions
    predictions = model.predict(processed_images)

    # Return the predictions
    response = {
        "statusCode": 200,
        "body": json.dumps(predictions.tolist())
    }
    return response

def preprocess_images(images):
    # Implement image preprocessing as per your model's requirements
    images_array = []
    for img in images:
        img = np.array(img) / 255.0  # Normalize if needed
        img = np.reshape(img, (150, 150, 1))  # Reshape for your model
        images_array.append(img)
    return np.array(images_array)
