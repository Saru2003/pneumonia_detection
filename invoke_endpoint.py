import boto3
import json

# Initialize the SageMaker runtime client
runtime_client = boto3.client('sagemaker-runtime')

# Load the input from the JSON file
with open('input.json', 'r') as json_file:
    payload = json_file.read()

# Invoke the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName='pneumonia-detection-endpoint',
    ContentType='application/json',
    Body=payload
)

# Read the response
result = json.loads(response['Body'].read().decode())
print("Prediction result:", result)
