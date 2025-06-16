import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras import models,layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout, BatchNormalization
import cv2
import numpy as np
import matplotlib.pyplot as plt


img_size=256

# 1.LOAD DATASET
full_dataset = tf.keras.utils.image_dataset_from_directory( # To load the dataset from directory
    'C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\classification\\data',  # path        
    labels='inferred',   # automatically assign labels based on folder names
    label_mode='binary', # labels will be 0 or 1 (binary classification)
    image_size=(img_size, img_size), # resize each image to 224*224 pixels
    batch_size=32,       # dataset will be loaded in batches of 32 images at a time
    shuffle=True        # shuffle the images randomly every epoch 
    )


# 2.SPLIT DATASET INTO Train Dataset,Validation Dataset, and Test Dataset

# Assign size for each set - Training(70%), Validation(20%), Test(10%)
dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size  

# Use take and skip to split the dataset
train_dataset = full_dataset.take(train_size) #train_dataset = take 70% of full_dataset 
val_dataset = full_dataset.skip(train_size).take(val_size) #Val_dataset = skip first 70% of full dataset and take next 20% of full dataset)
test_dataset = full_dataset.skip(train_size+val_size).take(test_size) # test_dataset= skip first 90% of full_dataset and take next 10% of full_dataset

# 3.BUILD MODEL
model = models.Sequential() #Sequential - to connect one layer after the other in a simple feed forward network
model.add(tf.keras.Input(shape=(img_size, img_size, 3))) #input layer
model.add(Rescaling(1./255)) #normalize pixel values to the range [0,1]

model.add(Conv2D(16, (3, 3), activation='relu')) #No.of filters = 16, Kernel size = 3*3, Activation function = Relu
model.add(MaxPooling2D((2, 2))) #kernel size = 2*2

model.add(Conv2D(32, (3, 3), activation='relu')) #No.of filters = 32, Kernel size = 3*3, Activation function = Relu
model.add(BatchNormalization()) #Activations(output of each layer) are maintained in a stable range by batch normalization layer for faster training
model.add(MaxPooling2D((2, 2))) #kernel size = 2*2
model.add(Dropout(0.1)) # Used to avoid overfitting

model.add(Flatten()) #Convert to 1D vector
model.add(Dense(128, activation='relu')) #Fully connected layer - no.of neurons =128, activation function = relu
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) #Fully connected layer - no.of neurons = 1, activation function = sigmoid (1 output for binary classification)



# 4.COMPILE MODEL
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5.TRAIN MODEL
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15
)

# 6.EVALUATE ON TEST DATASET
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"test_loss = {test_loss:.4f}")
print(f"test_accuracy = {test_accuracy:.4f}")

model.save("C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\classification\\model.h5")
print("model_1.h5")


# 7.Plot training and validation accuracy/loss curves
plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Display a few predicted images from test dataset
class_names = ['auto', 'pedestrian']  # 0 = auto, 1 = pedestrian

auto_images = []
pedestrian_images = []

for images, labels in test_dataset.take(10):  
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int).flatten()
    
    for i in range(len(predictions)):
        if predictions[i] == 0 and len(auto_images) < 3:
            auto_images.append((images[i].numpy(), class_names[0]))
        elif predictions[i] == 1 and len(pedestrian_images) < 3:
            pedestrian_images.append((images[i].numpy(), class_names[1]))
        
        if len(auto_images) == 3 and len(pedestrian_images) == 3:
            break
    if len(auto_images) == 3 and len(pedestrian_images) == 3:
        break

def plot_predicted_images(image_label_list, title):
    plt.figure(figsize=(12, 5))
    for i, (img, label) in enumerate(image_label_list):
        plt.subplot(1, 3, i+1)
        plt.imshow(img.astype("uint8"))
        plt.title(f"Predicted: {label}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_predicted_images(auto_images, "Predicted as Auto")
plot_predicted_images(pedestrian_images, "Predicted as Pedestrian")
