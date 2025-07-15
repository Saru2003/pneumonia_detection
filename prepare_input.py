import base64
import json

# Path to your image file
image_path = r"C:\Users\Asus\Documents\code\cloud_project\sample_pneumonia.jpeg"

# Read and encode the image
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Create input JSON with the correct key as expected by your model
input_json = {
    "instances": [
        {
            "data": image_base64  # Change 'image' to 'data' if that's what your model expects
        }
    ]
}

# Save to a JSON file
with open('input.json', 'w') as json_file:
    json.dump(input_json, json_file)

print("Input JSON created successfully!")
