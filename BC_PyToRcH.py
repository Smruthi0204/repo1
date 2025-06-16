import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn #nn.Module - base class from which class for building a model is derived
import torch.nn.functional as F
from torchsummary import summary #to print model summary
import matplotlib.pyplot as plt

image_size=120
batch_size=16

# 1.Pre-process and Load the Dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)), # Resize images 
    transforms.ToTensor(),  # Convert input image to tensors(input datastructure required by Pytorch for processing images in cnn)
                            # Normalization by scaling the pixel values to the range [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the pixel value range from [0,1] to [-1,1]
                                                                    # Pixel' = (pixel-mean)/std
])

dataset = datasets.ImageFolder(root='C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\classification\\data', transform=transform) #Load dataset (images with labels)

dataloader = DataLoader(dataset, batch_size, shuffle=True) # DataLoader- provides an iterator to load images in batches from dataset


# 2.Split Dataset into train_dataset, val_dataset, test_dataset (70:20:10) 
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) #Split dataset into 3 parts - train_dataset, val_dataset,
                                                                                         #test_dataset of size train_size, val_size, test_size respectively

# 3.Create dataloader for each set
train_loader = DataLoader(train_dataset, batch_size, shuffle=True) #DataLoader to load images in batches from train_dataset
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)    #DataLoader to load images in batches from val_dataset
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)  #DataLoader to load images in batches from test_dataset


# 4.Build Model
class Model_1(nn.Module):
    def __init__(self, num_classes):
        super(Model_1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (3,120,120) -> (32,120,120)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> (32,60,60)

            nn.Conv2d(32, 64, 3, padding=1),# -> (64,60,60)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> (64,30,30)

            nn.Conv2d(64, 128, 3, padding=1),# -> (128,30,30)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> (128,15,15)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 15 * 15, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # identify device as cpu or gpu
model = Model_1(num_classes=1).to(device) # num_classes=1 for binary classification (single output)
                                          # .to(device) moves the model's parameters to devic
summary(model, (3, image_size, image_size)) # display model summary                                          

# 5. Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for Binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer = adam
                                                           # learning rate = 0.001

# 6. Training 
train_losses = []
train_accuracies = []
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0 # running_loss - stores sum of all batch loss in one epoch.
                       # Used to compute average loss at the end of one epoch to monitor traning process
    correct_train=0
    total_train=0
    
    for images, labels in train_loader:
        images = images.to(device)  # Move images to device
        labels = labels.to(device).float().view(-1, 1)  # Move labels to device
                                                        # Convert to float which is required by the loss function BECLoss
                                                        # view(-1,1) - Reshape the label to match the shape of model's output

        optimizer.zero_grad()                                          
        outputs = model(images)                         # Pass the input batch through the model
        loss = criterion(outputs, labels)               # Calculate loss between predictions and labels using BECLoss function
        loss.backward()                                 # Performs back propogation to compute gradients of the loss with resepect to model parameters 
        optimizer.step()                                # Uodate model parameters using computed gradients

        running_loss += loss.item()                     # loss.item() converts loss tensor to float
        
        predicted = (outputs >= 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_train / total_train

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


# 7. Testing 
model.eval()
test_loss = 0.0
correct = 0 # counts the total number of correct predictions
total = 0   # counts the total number of samples evaluated

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device) 
        labels = labels.to(device).float().view(-1, 1) 

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predicted = (outputs >= 0.5).float()  # Threshold at 0.5
                                              # In a batch check each output value stored in the variable outputs and if value >= 0.5 convert to 1.0, else 0.0
        correct += (predicted == labels).sum().item() # Compare predicted values with labels
                                                      # Number of ones(True values) represent the number of correctly predicted values, so sum to compute the number of correctly predicted values
                                                      # .item() converts tensor to python float
        total += labels.size(0) # Compute the total samples evaluated

average_test_loss = test_loss / len(test_loader) # sum of loss of all batches / total number of batches
test_accuracy = 100 * correct / total # correctly predicted values / total predictions

print(f"\nTest Loss: {average_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

torch.save(Model_1,'C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\classification\\Model_1.pth')

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

import torchvision.utils as vutils

model.eval()
class_names = dataset.classes  # e.g., ['auto', 'pedestrian']
shown = {0: 0, 1: 0}
max_images = 3
shown_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predicted = (outputs >= 0.5).float().cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        images = images.cpu()

        for img, pred, label in zip(images, predicted, labels):
            if shown[int(pred)] < max_images:
                shown[int(pred)] += 1
                shown_images.append((img, int(pred)))
            if all(x == max_images for x in shown.values()):
                break
        if all(x == max_images for x in shown.values()):
            break

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for i, (img, label) in enumerate(shown_images):
    ax = axes[i // 3, i % 3]
    img = img * 0.5 + 0.5  # Unnormalize [-1,1] -> [0,1]
    npimg = img.permute(1, 2, 0).numpy()
    ax.imshow(npimg)
    ax.axis("off")
    ax.set_title(f"Predicted: {'Auto' if label==0 else 'Pedestrian'}")
plt.tight_layout()
plt.show()
