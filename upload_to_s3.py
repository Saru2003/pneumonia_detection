# import boto3

# s3_client = boto3.client('s3')

# def upload_to_s3(file_path, bucket_name, s3_file_name):
#     s3_client.upload_file(file_path, bucket_name, s3_file_name)

# # Example usage
# upload_to_s3(r'C:\Users\Asus\Documents\code\cloud_project\sample_pneumonia_2.jpeg', 'pneumonia-detection-model-bucket', 'image.jpeg')


import boto3
import json

# Initialize S3 client
s3_client = boto3.client('s3')

# Function to upload image to S3
def upload_to_s3(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name.split('/')[-1]
    
    # Upload the file
    s3_client.upload_file(file_name, bucket_name, object_name)
    print(f"File uploaded to s3://{bucket_name}/{object_name}")
    return f"s3://{bucket_name}/{object_name}"

# Function to invoke the SageMaker endpoint
def invoke_endpoint(endpoint_name, s3_uri):
    runtime_client = boto3.client('sagemaker-runtime')
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({'instances': [{'s3uri': s3_uri}]})  # Adjust based on the expected model input
    )
    return response['Body'].read().decode()

# Upload image to S3
bucket_name = 'pneumonia-detection-model-bucket'
file_path = r'C:\Users\Asus\Documents\code\cloud_project\sample_pneumonia_2.jpeg'  # Local file path
s3_uri = upload_to_s3(file_path, bucket_name)

# Invoke SageMaker endpoint with S3 URI
endpoint_name = 'pneumonia-detection-endpoint'
response = invoke_endpoint(endpoint_name, s3_uri)
print("Inference result:", response)
