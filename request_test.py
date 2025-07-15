# import requests
# import json
# import base64

# # Path to the image you want to send
# image_path = "C:\\Users\\Asus\\Documents\\code\\cloud_project\\sample_pneumonia.jpeg"

# # Read the image and encode it to base64
# with open(image_path, "rb") as f:
#     img_bytes = f.read()

# img_base64 = base64.b64encode(img_bytes).decode('utf-8')

# # Prepare the payload
# payload = json.dumps({"images": [img_base64]})  # Adjust as necessary for your model

# # Specify the endpoint URL
# endpoint_url = "https://runtime.sagemaker.eu-north-1.amazonaws.com/endpoints/pneumonia-detection-endpoint/invocations"

# # Send the POST request
# response = requests.post(endpoint_url, data=payload, headers={"Content-Type": "application/json"})

# # Print the response
# print("Response Status Code:", response.status_code)
# print("Response Body:", response.json())

import boto3
import json
import base64

# Initialize the boto3 client
runtime_client = boto3.client('sagemaker-runtime', region_name='eu-north-1')

# Path to the image you want to send
image_path = "C:\\Users\\Asus\\Documents\\code\\cloud_project\\sample_pneumonia.jpeg"

# Read the image and encode it to base64
with open(image_path, "rb") as f:
    img_bytes = f.read()

img_base64 = base64.b64encode(img_bytes).decode('utf-8')

# Prepare the payload
# Prepare the payload
payload = json.dumps({"data": [img_base64]})  # Change 'images' to 'data' or whatever your model expects

# Invoke the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName='pneumonia-detection-endpoint',
    ContentType='application/json',
    Body=payload
)

# Print the response
response_body = response['Body'].read().decode('utf-8')
print("Response Body:", response_body)
