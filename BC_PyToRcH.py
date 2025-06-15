import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn #nn.Module - base class from which class for building a model is derived
import torch.nn.functional as F

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

dataset = datasets.ImageFolder(root='D:\\OneDrive - TVS Motor Company Ltd\\Desktop\\dataset', transform=transform) #Load dataset (images with labels)

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
                                                      # (no.of channels, height, width)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)  # input = (3, 120, 120),  output = (16, 118, 118)
        self.pool1 = nn.MaxPool2d(2, 2)               # input = (16, 118, 118), output = (16, 59, 59)                                                  

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # input = (16, 59, 59), output = (32, 57, 57)
        self.pool2 = nn.MaxPool2d(2, 2)               # input = (32, 57, 57), output = (32, 28, 28)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3) # input = (32, 28, 28), output = (64, 26, 26)
        self.pool3 = nn.MaxPool2d(2, 2)               # input = (64, 26, 26), output = (64, 13, 13)
        
        self.fc1 = nn.Linear(64*13*13, 128)           # input = (64, 13, 13)
        self.fc2 = nn.Linear(128, 32)                 # input = 128, output = 32
        self.fc3 = nn.Linear(32,1)                    # input = 32, output = 1 (binary classification)
        


    def forward(self, x):   # Forward pass function which takes x (batch of input images) as input and returns the network's output
        x = self.pool1(F.relu(self.conv1(x))) 
        x = self.pool2(F.relu(self.conv2(x)))    # Pass the input through each layer of the network with Relu activation function
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 13 * 13)  # Flatten to 1D vector before feeding to fully connected layer
                                      # -1 is used to automatically infer the dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))       # Fully connected layer with Relu Activation function
        x = torch.sigmoid(self.fc3(x)) # Output layer with Sigmoid Activation function 

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # identify device as cpu or gpu
model = Model_1(num_classes=1).to(device) # num_classes=1 for binary classification (single output)
                                          # .to(device) moves the model's parameters to device

# 5. Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for Binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer = adam
                                                           # learning rate = 0.001

# 6. Training 
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0 # running_loss - stores sum of all batch loss in one epoch.
                       # Used to compute average loss at the end of one epoch to monitor traning process

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


torch.save(Model_1,'D:\\OneDrive - TVS Motor Company Ltd\\Desktop\\Model_1.pth')
