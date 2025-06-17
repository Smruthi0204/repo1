import cv2
import numpy as np
from tensorflow import keras

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    img = cv2.resize(img, (256,256))  # Resize to 256x256 (width, height)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load the model
model = keras.models.load_model("D:\\OneDrive - TVS Motor Company Ltd\\Desktop\\model.h5")

# Test image path
test_image_path1 = "D:\\OneDrive - TVS Motor Company Ltd\\Desktop\\img1.jpeg"
test_image_path2 = "D:\\OneDrive - TVS Motor Company Ltd\\Desktop\\img4.jpg"
test_image_path3 = "D:\\OneDrive - TVS Motor Company Ltd\\Desktop\\img3.jpeg"


# Preprocess image
preprocessed_image1 = preprocess_image(test_image_path1)
preprocessed_image2 = preprocess_image(test_image_path2)
preprocessed_image3 = preprocess_image(test_image_path3)

# Predict
prediction1 = model.predict(preprocessed_image1)
prediction2 = model.predict(preprocessed_image2)
prediction3 = model.predict(preprocessed_image3)

print("Prediction1:\n",prediction1)
print("Prediction2:\n",prediction2)
print("Prediction3:\n",prediction3)



