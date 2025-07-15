# import boto3
# import json
# import numpy as np

# # Create SageMaker runtime client
# runtime_client = boto3.client('sagemaker-runtime', region_name='eu-north-1')

# # Create a sample image array for testing
# # Here, we'll create a dummy grayscale 150x150 image
# sample_image = np.zeros((150, 150, 1)).tolist()  # Replace with your actual image array
# payload = {
#     "instances": [{"image": sample_image}]  # Wrap the image array in a list
# }

# try:
#     # Invoke the endpoint
#     response = runtime_client.invoke_endpoint(
#         EndpointName='pneumonia-detection-endpoint',
#         ContentType='application/json',
#         Body=json.dumps(payload)
#     )

#     print("Response Status Code:", response['ResponseMetadata']['HTTPStatusCode'])
#     print("Response Body:", response['Body'].read().decode())
# except Exception as e:
#     print("Error:", str(e))

# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load the model
# model = load_model('pneumonia_detection_saved_model/1')

# # Load and preprocess the image
# image_path = 'path_to_your_image.jpg'  # Update with your image path
# image = Image.open(image_path)

# # Convert to grayscale if needed
# if image.mode != 'L':
#     image = image.convert('L')

# # Resize and normalize the image
# image = image.resize((150, 150))
# image_array = np.array(image) / 255.0  # Normalize if needed
# image_array = image_array.reshape((1, 150, 150, 1))  # Reshape for your model

# # Make prediction
# prediction = model.predict(image_array)
# print(f"Prediction: {prediction}")

import boto3
import json
import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('L')  
    image = image.resize((150, 150))  
    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0)  
    image_np = np.expand_dims(image_np, axis=3)  
    return image_np.tolist()  

image_path = r'C:/Users/Asus/Documents/code/cloud_project/sample_normal.jpeg'
image_data = load_image(image_path)
runtime_client = boto3.client('sagemaker-runtime', region_name='eu-north-1')

response = runtime_client.invoke_endpoint(
    EndpointName='pneumonia-detection-endpoint-v2',
    ContentType='application/json',
    Body=json.dumps({"instances": image_data})
)
result = json.loads(response['Body'].read().decode())
print("Prediction result:", result)
