import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


model = keras.models.load_model("D:\\project1\\image_classification\\model.h5")

test_image1 = "D:\\project1\\image_classification\\pedestrian&auto1.jpg"
test_image2 = "D:\\project1\\image_classification\\pedestrian&auto2.jpg"
test_image3 = "D:\\project1\\image_classification\\pedestrian&auto3.jpg"
 
preprocessed_image1 = preprocess_image(test_image1)
preprocessed_image2 = preprocess_image(test_image2)
preprocessed_image3 = preprocess_image(test_image3)

prediction1 = model.predict(preprocessed_image1)
p1 = (prediction1 > 0.5).astype(int)
prediction2 = model.predict(preprocessed_image2)
p2 = (prediction2> 0.5).astype(int)
prediction3 = model.predict(preprocessed_image3)
p3 = (prediction3 > 0.5).astype(int)

print("Prediction1:\n",p1)
print(prediction1)
img1= cv2.imread(test_image1)
plt.imshow(img1)
plt.title("image 1")
plt.show()

print("Prediction2:\n",p2)
print(prediction2)
img2= cv2.imread(test_image2)
plt.imshow(img2)
plt.title("image 2")
plt.show()

print("Prediction3:\n",p3)
print(prediction3)
img3= cv2.imread(test_image3)
plt.imshow(img3)
plt.title("image 3")
plt.show()
