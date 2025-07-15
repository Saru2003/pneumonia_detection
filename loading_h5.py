# from keras.models import load_model
# import cv2
# import numpy as np

# # Load the saved model
# model = load_model('pneumonia_detection_model.h5')

# # Function to preprocess the image
# def preprocess_image(img_path, img_size=150):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
#     img = cv2.resize(img, (img_size, img_size))  # Resize to the input size of the model
#     img = img / 255.0  # Normalize the image
#     img = img.reshape(1, img_size, img_size, 1)  # Reshape to match the input shape of the model
#     return img

# # Path to the image you want to predict
# img_path = r"C:\Users\Asus\Documents\code\cloud_project\archive\chest_xray\test\NORMAL\NORMAL2-IM-0259-0001.jpeg"

# # Preprocess the image
# preprocessed_img = preprocess_image(img_path)

# # Predict the class of the image
# prediction = model.predict(preprocessed_img)

# # Convert the prediction probability into a class label
# predicted_class = int(prediction[0][0] > 0.5)  # Assuming 0 = Pneumonia, 1 = Normal

# labels = ['Pneumonia', 'Normal']
# print(f'The predicted class is: {labels[predicted_class]}')
from keras.models import load_model
import cv2
import numpy as np

model = load_model('pneumonia_detection_model.h5')
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

# preprocess input image
def prepare_image(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
        resized_img = cv2.resize(img, (img_size, img_size))
        normalized_img = resized_img / 255.0  
        reshaped_img = normalized_img.reshape(-1, img_size, img_size, 1)  
        return reshaped_img
    except Exception as e:
        print(f"Error processing image {filepath}: {e}")
        return None

image_path = 'sample_pneumonia_2.jpeg'
image = prepare_image(image_path)

if image is not None:
    prediction = model.predict(image)
    prediction_label = labels[int(prediction > 0.5)]  # 0: PNEUMONIA, 1: NORMAL
    print(f"The predicted label is: {prediction_label}")
else:
    print("Failed to process the image.")
