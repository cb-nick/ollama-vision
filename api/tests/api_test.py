import base64
import requests

# Path to the image file
import os
image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")

# Encode the image as base64
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Define the API endpoint
url = "http://localhost:8505/analyze-image-base64"

# Create the payload
payload = {"image_base64": image_base64}

# Make the POST request
response = requests.post(url, json=payload)

# Print the response
print(response.json())