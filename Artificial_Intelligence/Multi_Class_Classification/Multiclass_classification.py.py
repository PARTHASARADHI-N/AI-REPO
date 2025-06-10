import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import WeightedRandomSampler
import os
import cv2
import copy
import sklearn
from sklearn.metrics import f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
        self.cnn1 = nn.Conv2d(3, 48, kernel_size=7, stride=4) 
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cnn2 = nn.Conv2d(48, 128, kernel_size=3, padding=1) 
        self.cnn3 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.cnn4 = nn.Conv2d(192, 192, kernel_size=3)
        self.cnn5 = nn.Conv2d(192, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 8 * 8, 2048) 
        self.fc2 = nn.Linear(2048, 1000)
        self.fc3 = nn.Linear(1000, num_classes) 


    def forward(self, x):
        x = self.pool1(F.relu(self.cnn1(x)))
        x = self.pool2(F.relu(self.cnn2(x)))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        x = F.relu(self.cnn5(x))


        # Flatten for FC layer
        x = x.view(-1, 128 * 8 * 8) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_model(model, test_loader, test_dataset, device, output_csv='all_2048.csv'):
    model.eval()
    results = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)  

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

           
            image_path, _ = test_dataset.samples[idx] 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            results.append((os.path.basename(image_path), predicted.cpu().numpy()[0], labels.cpu().numpy()))  
  
    
    
  
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label']) 
        for image_name, predictedlabel, actual in results:
            writer.writerow([predictedlabel]) 

    print(f"Results saved to {output_csv}")

def test_single_image(image_path, model, transform, class_names):
    print("Testing single image...")
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 
    image = Variable(image).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    print(f"Predicted Class Index: {predicted.item()}")
    print(f"Predicted Class Name: {class_names[predicted.item()]}")
   
    

def main():
    # Command line arguments
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3]

    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(saturation=(0.7, 1.3)),
        transforms.ToTensor(),
    ])
    transform1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    dataset = datasets.ImageFolder(root=dataPath, transform=transform)
    class_names = dataset.classes 
    num_classes = len(class_names) 
    print("Class names:", class_names)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    
    train_labels = []
    sample_weights = []
    for index in train_dataset.indices:
         train_labels.append(dataset.targets[index])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights_tensor = torch.from_numpy(class_weights).float().to(device) 

    for label in train_labels:
        sample_weights.append(class_weights[label])

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BirdClassifier(num_classes=num_classes).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
   
    best_loss = float('inf')
    best_model_weights = None
    patience = 3
    min_delta = 0.001
    if trainStatus == 'train':
        print("Starting training...")
        training_loss=[]
        validation_loss=[]
        for epoch in range(20): 
 
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
  
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch + 1}/20], Batch [{batch_idx}], Loss: {loss.item():.4f}")

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            print(f"Epoch [{epoch + 1}/20] Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
            training_loss.append(round(train_accuracy,2))

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
 
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            print(f"Epoch [{epoch + 1}/20] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            validation_loss.append(round(val_accuracy,2))
   
            if(val_loss<best_loss-min_delta):
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = 3
                model.load_state_dict(best_model_weights)
                torch.save(model.state_dict(), modelPath)
                print("Model saved.")
            else:
                patience -= 1
                if patience == 0:
                   print("Early stopping triggered.")
                   break

        print("training_loss :",training_loss)
        print("validation_loss :",validation_loss)

    model.load_state_dict(torch.load(modelPath))
    model.eval()

    if trainStatus == 'single_test':
        image_path = sys.argv[4]
        test_single_image(image_path, model, transform1, class_names)

    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.to(device)
    if trainStatus == 'test':
        test_dataset = datasets.ImageFolder(root=dataPath, transform=transform1)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_model(model, test_loader,test_dataset, device)  
   

if __name__ == "__main__":
    main()
