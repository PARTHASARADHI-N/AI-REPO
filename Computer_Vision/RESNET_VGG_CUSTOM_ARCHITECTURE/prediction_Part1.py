import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


def training_code_vgg():
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    import h5py
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Set random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define transformations with Color Jittering
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    # Define dataset class
    class PrepareData(Dataset):
        def __init__(self, datapath, labelpath, transform=None):
            self.datapath = datapath
            self.labelpath = labelpath
            self.transform = transform

            with h5py.File(self.datapath, "r") as D:
                self.images = D["x"][:]  # Load images
            with h5py.File(self.labelpath, "r") as L:
                self.labels = L["y"][:]  # Load labels (binary: 0 or 1)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            label = torch.tensor(label, dtype=torch.long).squeeze()
            if self.transform:
                image = self.transform(image)
            return image, label

    # Define file paths
    train_datafile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_train_x.h5"
    train_labelfile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_train_y.h5"
    val_datafile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_valid_x.h5"
    val_labelfile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_valid_y.h5"
    test_datafile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_test_x.h5"
    test_labelfile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_test_y.h5"
    # Create datasets and loaders
    train_dataset = PrepareData(train_datafile, train_labelfile, transform=train_transform)
    val_dataset = PrepareData(val_datafile, val_labelfile)
    test_dataset = PrepareData(test_datafile, test_labelfile)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained VGG16
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 2)
    for param in vgg16.parameters():
        param.requires_grad = True
    vgg16 = vgg16.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 25
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        vgg16.train()
        running_loss, all_preds, all_labels = 0.0, [], []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        
        vgg16.eval()
        running_val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = vgg16(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    # Save model
    torch.save(vgg16.state_dict(), "vgg16_pcam_CEL_ADAM_0.0001_data_aug.pth")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, label="Train Accuracy", marker='o')
    plt.plot(range(1, num_epochs+1), val_accs, label="Validation Accuracy", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_plot.png")

    plt.show()

    # Testing phase
    vgg16.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels)

    # Compute test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)

    print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")
def training_code_resnet():
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    import h5py
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from torch.optim.lr_scheduler import StepLR

    # Set random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define dataset class
    class PrepareData(Dataset):
        def __init__(self, datapath, labelpath, transform=None):
            self.datapath = datapath
            self.labelpath = labelpath
            self.transform = transform

            with h5py.File(self.datapath, "r") as D:
                self.images = D["x"][:]  # Load images
            with h5py.File(self.labelpath, "r") as L:
                self.labels = L["y"][:]  # Load labels (binary: 0 or 1)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            label = torch.tensor(label, dtype=torch.long).squeeze()
            if self.transform:
                image = self.transform(image)
            return image, label

    # Define data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    # Define file paths
    train_datafile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_train_x.h5"
    train_labelfile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_train_y.h5"
    val_datafile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_valid_x.h5"
    val_labelfile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_valid_y.h5"
    test_datafile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_test_x.h5"
    test_labelfile = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_test_y.h5"

    # Create datasets and loaders with augmentation for training
    train_dataset = PrepareData(train_datafile, train_labelfile, transform=train_transform)
    val_dataset = PrepareData(val_datafile, val_labelfile)
    test_dataset = PrepareData(test_datafile, test_labelfile)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained resnet50
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 2)
    for param in resnet50.parameters():
        param.requires_grad = True
    resnet50 = resnet50.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.parameters(), lr=0.00001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # Training loop
    num_epochs = 25
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(num_epochs):
        resnet50.train()
        running_loss, all_preds, all_labels = 0.0, [], []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        scheduler.step()

        # Training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        # Validation metrics
        resnet50.eval()
        running_val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet50(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    # Save model
    torch.save(resnet50.state_dict(), "resnet50_CEL_ADAM_0.00001_data_aug_SLR.pth")

    # Plot and save metrics
    def save_plot(train_vals, val_vals, ylabel, filename):
        plt.figure(figsize=(10,5))
        plt.plot(train_vals, label=f'Train {ylabel}')
        plt.plot(val_vals, label=f'Validation {ylabel}')
        plt.title(f'{ylabel} per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    save_plot(train_losses, val_losses, 'Loss', 'loss_plot.png')
    save_plot(train_accs, val_accs, 'Accuracy', 'accuracy_plot.png')
    save_plot(train_precisions, val_precisions, 'Precision', 'precision_plot.png')
    save_plot(train_recalls, val_recalls, 'Recall', 'recall_plot.png')
    save_plot(train_f1s, val_f1s, 'F1 Score', 'f1_plot.png')

    # Testing phase
    resnet50.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet50(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels)

    # Compute test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)

    print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((96, 96)),  
    transforms.ToTensor(),

])

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx] 

def load_model(model_type):
    if model_type.lower() == "resnet":
        model = models.resnet50(weights=None) 
        model.fc = nn.Linear(model.fc.in_features, 2)  
        model_path = "resnet50_pcam_CEL_ADAM_SLR_0.00001_data_aug.pth"
    elif model_type.lower() == "vgg":
     
        model = models.vgg16(weights=None)  
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2) 
        model_path = "vgg16_pcam_CEL_ADAM_0.0001_data_aug.pth"
    else:
        raise ValueError("Model type must be 'resnet' or 'vgg'")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def main():

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <model_type>")
        print("model_type must be 'resnet' or 'vgg'")
        sys.exit(1)

    input_folder = sys.argv[1]
    model_type = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        sys.exit(1)

    try:
        model = load_model(model_type)
    except ValueError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Model file for {model_type} not found")
        sys.exit(1)

    dataset = ImageFolderDataset(input_folder, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)  
    predictions = []
    image_names = []

    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            image_names.extend(names)

    # Save results to result.txt
    with open("result.txt", "w") as f:
        f.write("Image Name,Prediction\n")
        for name, pred in zip(image_names, predictions):
            if pred==1:
               pred="positive"
               f.write(f"{name},{pred}\n")
            elif pred==0:
               pred="negative"
               f.write(f"{name},{pred}\n")

    print(f"Predictions saved to result.txt using {model_type} model")
    print("Metrics (accuracy, precision, recall, F1-score) require ground truth labels, which were not provided.")

if __name__ == "__main__":
    main()