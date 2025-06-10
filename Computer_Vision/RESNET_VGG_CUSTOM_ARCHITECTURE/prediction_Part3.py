import os
import sys
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from torch.optim.lr_scheduler import StepLR
from PIL import Image

# Set seeds for consistent results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def training_code():
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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import time
    from torch.optim.lr_scheduler import StepLR

    # Set seeds for consistent results
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Depthwise separable convolution layer
    class DepthwiseSeparableLayer(nn.Module):
        def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
            super().__init__()
            self.depth = nn.Conv2d(in_c, in_c, kernel_size=kernel_size,
                                stride=stride, padding=padding, groups=in_c)
            self.point = nn.Conv2d(in_c, out_c, kernel_size=1)
            

        def forward(self, x):
            x = self.depth(x)
            x = self.point(x)
            return x

    # Residual block with depthwise separable convolutions
    class ResidualBlock(nn.Module):
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.conv1 = DepthwiseSeparableLayer(in_c, out_c, kernel_size=3,
                                                stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = DepthwiseSeparableLayer(out_c, out_c, kernel_size=3,
                                                padding=1)
            self.bn2 = nn.BatchNorm2d(out_c)

            self.skip = nn.Sequential()
            if in_c != out_c or stride != 1:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c)
                )

        def forward(self, x):
            res = self.skip(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x += res
            x = self.relu(x)
            return x

    # Custom model architecture
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            # First convolution block
            self.conv_block = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # Residual blocks
            self.residual_blocks = nn.Sequential(
                ResidualBlock(64, 64, stride=1),
                ResidualBlock(64, 128, stride=2),
                ResidualBlock(128, 256, stride=2)
            )
            # Inception module (without nn.Sequential)
            self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
            self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(256, 32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.bn = nn.BatchNorm2d(224)  # 64 + 128 + 32 = 224
            # Pooling and dense layers
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(224, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            self.output = nn.Linear(128, 1)

        def forward(self, x):
            x = self.conv_block(x)
            x = self.residual_blocks(x)
            # Inception module
            b1 = self.conv1(x)
            b1 = self.bn1(b1)
            b1 = self.relu(b1)
            b2 = self.conv2(x)
            b2 = self.bn2(b2)
            b2 = self.relu(b2)
            b3 = self.conv3(x)
            b3 = self.bn3(b3)
            b3 = self.relu(b3)
            x = torch.cat([b1, b2, b3], dim=1)
            x = self.bn(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return self.output(x)

   
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
            label = torch.tensor(label, dtype=torch.float32).squeeze()
            if self.transform:
                image = self.transform(image)
            return image, label


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    # Data file paths
    train_data = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_train_x.h5"
    train_labels = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_train_y.h5"
    val_data = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_valid_x.h5"
    val_labels = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_valid_y.h5"
    test_data = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_test_x.h5"
    test_labels = "/kaggle/input/pcam-data-raw/camelyonpatch_level_2_split_test_y.h5"

    # Create datasets and loaders
    train_set = PrepareData(train_data, train_labels, transform=train_transform)
    val_set = PrepareData(val_data, val_labels)
    test_set = PrepareData(test_data, test_labels)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    # Setup model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel().to(device)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=0.0001)
    # opt = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    # Show parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {params} trainable parameters")

    # Training loop
    epochs = 25
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        loss_sum = 0.0
        preds_list = []
        labels_list = []
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs).squeeze()
                loss = loss_fn(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
            lbls = lbls.cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(lbls)
        scheduler.step()
        
        train_loss = loss_sum / len(train_loader)
        train_acc = accuracy_score(labels_list, preds_list)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                out = model(imgs).squeeze()
                loss = loss_fn(out, lbls)
                val_loss_sum += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
                lbls = lbls.cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(lbls)
        val_loss = val_loss_sum / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        end = time.time()
        time_taken = (end - start) / 60
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {time_taken:.2f} mins")

    # Save the model
    torch.save(model.state_dict(), "model_0.001_data_aug_sgd.pth")

    # Test the model (with fix)
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            out = model(imgs).squeeze()
            # Ensure out is a 1D tensor
            if out.dim() == 0:
                out = out.unsqueeze(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            # Ensure lbls is a 1D tensor
            if lbls.dim() == 0:
                lbls = lbls.unsqueeze(0)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(lbls.cpu().numpy())

    # Convert to numpy arrays
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    # Test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds)
    test_rec = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    conf_matrix = confusion_matrix(test_labels, test_preds)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Train Loss', color='green')
    plt.plot(val_loss_list, label='Val Loss', color='red')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_graph.png")
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, label='Train Acc', color='green')
    plt.plot(val_acc_list, label='Val Acc', color='red')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("acc_graph.png")
    plt.show()

class DepthwiseSeparableLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depth = nn.Conv2d(in_c, in_c, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=in_c)
        self.point = nn.Conv2d(in_c, out_c, kernel_size=1)
        

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableLayer(in_c, out_c, kernel_size=3,
                                             stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableLayer(out_c, out_c, kernel_size=3,
                                             padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.skip = nn.Sequential()
        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = self.relu(x)
        return x


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )

        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(224)  # 64 + 128 + 32 = 224
        # Pooling and dense layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(224, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        # Inception module
        b1 = self.conv1(x)
        b1 = self.bn1(b1)
        b1 = self.relu(b1)
        b2 = self.conv2(x)
        b2 = self.bn2(b2)
        b2 = self.relu(b2)
        b3 = self.conv3(x)
        b3 = self.bn3(b3)
        b3 = self.relu(b3)
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.bn(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.output(x)

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
        label = torch.tensor(label, dtype=torch.float32).squeeze()
        if self.transform:
            image = self.transform(image)
        return image, label

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


test_transform = transforms.Compose([
    transforms.Resize((96, 96)),  
    transforms.ToTensor(),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomModel().to(device)
model_path = "Improved_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

def main():

    if len(sys.argv) != 2:
        print("Usage: python script.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        sys.exit(1)

    dataset = ImageFolderDataset(input_folder, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)  

    predictions = []
    image_names = []

    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()  
            predictions.extend(preds)
            image_names.extend(names)

    with open("result.txt", "w") as f:
        f.write("Image Name,Prediction\n")
        for name, pred in zip(image_names, predictions):
            if pred==1:
               pred="positive"
               f.write(f"{name},{pred}\n")
            elif pred==0:
               pred="negative"
               f.write(f"{name},{pred}\n")  

    print("Predictions saved to result.txt")
    print("Metrics (accuracy, precision, recall, F1-score) require ground truth labels, which were not provided.")

if __name__ == "__main__":
    main()