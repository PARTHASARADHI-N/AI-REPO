import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from collections import Counter

import csv
import copy
from scipy.stats import multivariate_normal
torch.manual_seed(0)
class NPZDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file, allow_pickle=True)  
        self.images = data['data']
        self.labels = data['labels']
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class GMM:
    def __init__(self, k, dim, initial_mu=None, initial_sigma=None, initial_weights=None):
        np.random.seed(0)
        self.k = k 
        self.dim = dim 
        self.mu = initial_mu if initial_mu is not None else np.random.rand(k, dim)
        self.sigma = initial_sigma if initial_sigma is not None else np.array([np.eye(dim) for _ in range(k)])
        self.weights = initial_weights if initial_weights is not None else np.ones(k) / k

    def likelihood(self, X):
        log_ll = 0
        for n in X:
            sum_prob = 0
            for i in range(self.k):
                sum_prob += self.weights[i] * multivariate_normal.pdf(n, mean=self.mu[i], cov=self.sigma[i])
            log_ll += np.log(sum_prob + 1e-10)
        return log_ll

    def Expectation(self, data):
        N = data.shape[0]
        self.zk = np.zeros((N, self.k))
        for i in range(self.k):
            self.zk[:, i] = self.weights[i] * multivariate_normal.pdf(data, mean=self.mu[i], cov=self.sigma[i])
        self.zk /= self.zk.sum(axis=1, keepdims=True) + 1e-10

    def Maximization(self, data):
        N, d = data.shape
        for i in range(self.k):
            Nk = np.sum(self.zk[:, i])
            self.weights[i] = Nk / N
            self.mu[i] = np.sum(self.zk[:, i, np.newaxis] * data, axis=0) / Nk
            diff = data - self.mu[i]
            self.sigma[i] = (self.zk[:, i, np.newaxis] * diff).T @ diff / Nk
            self.sigma[i] += np.eye(d) * 1e-6

    def fit(self, data, max_iter=50, tol=1e-3):
        prev_log_likelihood = None
        for iteration in range(max_iter):
            self.Expectation(data)
            self.Maximization(data)
            log_likelihood = self.likelihood(data)
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < tol:
                print(f"Converged at iteration {iteration}")
                break
            prev_log_likelihood = log_likelihood
        return self

    def predict(self, X):
        self.Expectation(X)
        return np.argmax(self.zk, axis=1)
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.l1 = nn.Linear(28*28, 392)
        self.l2 = nn.Linear(392, 196)
        self.l3 = nn.Linear(196, 98)
        self.l4 = nn.Linear(98, 8)
        self.mu = nn.Linear(8, 8)
        self.log_var = nn.Linear(8, 8)
        self.l5 = nn.Linear(8, 98)
        self.l6 = nn.Linear(98,196)
        self.l7 = nn.Linear(196,392)
        self.l8 = nn.Linear(392,28*28)
     
    def encode(self, x):
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = nn.ReLU()(self.l3(x))
        x = nn.ReLU()(self.l4(x))
        
        return x
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        x = nn.ReLU()(self.l5(z))
        x = nn.ReLU()(self.l6(x))
        x = nn.ReLU()(self.l7(x))                 
        x = torch.sigmoid(self.l8(x))
        return x

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded_result = self.encode(x)
        MU = self.mu(encoded_result)
        SIGMA=self.log_var(encoded_result)
        reparam=self.reparameterize(MU,SIGMA)
        decoded_result = self.decode(reparam)
        return encoded_result, decoded_result,MU,SIGMA
    
def evaluate_gmm_performance(labels_true, labels_pred):
    accuracy = accuracy_score(labels_true, labels_pred)
    precision_macro = precision_score(labels_true, labels_pred, average='macro') 
    recall_macro = recall_score(labels_true, labels_pred, average='macro') 
    f1_macro = f1_score(labels_true, labels_pred, average='macro') 
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 3*KLD
def save_gmm_and_labels(gmm, cluster_to_label, filename='gmm_params.pkl'):
    gmm_and_labels = {
        'gmm': gmm,
        'cluster_to_label': cluster_to_label
    }
    with open(filename, 'wb') as f:
        pickle.dump(gmm_and_labels, f)
    print(f"GMM and cluster labels saved to {filename}")
def load_gmm_and_labels(filename='gmm_params.pkl'):
    with open(filename, 'rb') as f:
        gmm_and_labels = pickle.load(f)
    gmm = gmm_and_labels['gmm']
    cluster_to_label = gmm_and_labels['cluster_to_label']

    return gmm, cluster_to_label
def train_phase(model, train_loader, val_loader,save_path, gmm_save_path,device,epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss(reduction="sum")
    print(len(train_loader))
    latent_vectors = []
    data_loader = DataLoader(dataset=train_loader.dataset, batch_size=32, shuffle=False)
    val_data_loader=DataLoader(dataset=val_loader.dataset, shuffle=False)
    for epoch in range(epochs):
        total_loss = 0.0
        latent_vectors = []
        best_loss = float('inf')
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.view(data.size(0), -1).to(device)
            encoded, decoded, mu, log_var = model(data)
            latent_vectors.append(mu.cpu().detach().numpy())
            loss = loss_function(decoded,data,mu,log_var)
            if(loss<best_loss):
                best_loss=loss
                best_model_weights = copy.deepcopy(model.state_dict())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epoch+1}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
            total_loss += loss.item()
        model.load_state_dict(best_model_weights)
        epoch_loss = total_loss / len(data_loader)
        print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1,epochs, epoch_loss)
        )
    torch.save(model.state_dict(), save_path)
    latent_vectors = np.concatenate(latent_vectors)
    gmm=GMM(3,8)
    gmm.fit(latent_vectors)
   
    true_labels = []
    val_latent_vectors = []
    with torch.no_grad():
        for data, labels in val_data_loader:
            data = data.view(data.size(0), -1).to(device)
            encoded_result, _, Mu, sigma = model(data)
            true_labels.append(labels.cpu().numpy())
            val_latent_vectors.append(Mu.cpu().numpy())

    val_latent_vectors = np.concatenate(val_latent_vectors)
    cluster_predictions = gmm.predict(val_latent_vectors)
    true_labels = np.array(true_labels) 

    cluster_to_label = {}
    for cluster_id in np.unique(cluster_predictions):
        cluster_indices = np.where(cluster_predictions == cluster_id)[0]
        cluster_labels = true_labels[cluster_indices]
        cluster_labels = cluster_labels.flatten()
        majority_label = Counter(cluster_labels).most_common(1)[0][0]
        cluster_to_label[cluster_id] = majority_label
    save_gmm_and_labels(gmm, cluster_to_label,gmm_save_path)

def test_reconstruction(model, test_loader):
    model.eval()
    all_recons = []
    all_orig = [] 
    true_labels = [] 
    gmm_pred = []
    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1).to(device)
            encoded, recon, mu, log_var = model(data)
            
            all_recons.append(recon.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(log_var.cpu().numpy()) 
            
            all_orig.append(data.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    np.savez("vae_reconstructed.npz", reconstructions=np.concatenate(all_recons))
    all_recons = np.concatenate(all_recons)
    all_orig = np.concatenate(all_orig)
    num_images = len(all_orig)  
    plt.figure(figsize=(15, num_images))

    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(all_orig[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(2, num_images, i+1+num_images)
        plt.imshow(all_recons[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    gmm, cluster_to_label = load_gmm_and_labels('gmm_params.pkl')
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1).to(device)
            encoded_result, _,Mu,sigma = model(data)
            predictions = gmm.predict(Mu.cpu().numpy())
            
            gmm_pred.extend(predictions)
    new_labeled_predictions = [cluster_to_label[cluster] for cluster in gmm_pred]

    print((true_labels),(gmm_pred))
    print("Length of true labels:", len(true_labels))
    print("Length of predicted labels:", len(new_labeled_predictions))
    print(true_labels,new_labeled_predictions)

    metrics = evaluate_gmm_performance(true_labels, new_labeled_predictions)
    print("GMM Performance Metrics:")
    print(metrics)

def test_classifier(model, test_loader, gmm_params_path):
    model.eval()
    true_labels = []
    gmm_pred = []
    latents=[]
    lab=[]
    gmm, cluster_to_label = load_gmm_and_labels(gmm_params_path)
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1).to(device)
            encoded_result, _,mu,sigma = model(data)
            predictions = gmm.predict(mu.cpu().numpy())
            gmm_pred.extend(predictions)
            true_labels.extend(labels.numpy()) 
            latents.append(mu.cpu().numpy()) 
            lab.extend(np.atleast_1d(predictions))
            
        # print(true_labels,gmm_pred)
    new_labeled_predictions = [cluster_to_label[cluster] for cluster in gmm_pred]
    latents = np.concatenate(latents, axis=0) 
    lab = np.array(lab)
    
    # pca = PCA(n_components=2)
    # latents_2d = pca.fit_transform(latents)

    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=lab, cmap='jet', alpha=0.5)
    # plt.colorbar(scatter) 
    # plt.title("2D Scatter Plot of Latent Vectors (Digits 1, 4, and 8)")
    # plt.xlabel("Latent Dim 1")
    # plt.ylabel("Latent Dim 2")
    # plt.show()
    # print(true_labels,new_labeled_predictions)
    metrics = evaluate_gmm_performance(true_labels, new_labeled_predictions)
    print("GMM Classifier Performance Metrics:")
    print(metrics)
    with open("vae.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label']) 
        for  i in range(len(new_labeled_predictions)):
            writer.writerow([new_labeled_predictions[i]]) 
         

    print("Results saved to vae.csv")

# Main logic for script execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Determine mode based on input arguments
    if len(sys.argv) >= 5 and sys.argv[3] == "train":
        train_data_path = sys.argv[1]
        val_data_path = sys.argv[2]
        model_save_path = sys.argv[4]
        gmm_save_path = sys.argv[5]

        train_dataset = NPZDataset(train_data_path, transform=transform)
        val_dataset = NPZDataset(val_data_path, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, shuffle=False)

        model = VAE().to(device)
        print("Starting training...")
        train_phase(model, train_loader,val_loader, model_save_path, gmm_save_path, device,100)
        print("Training completed.")

    elif len(sys.argv) >= 4 and sys.argv[2] == "test_reconstruction":
        test_data_path = sys.argv[1]
        model_load_path = sys.argv[3]

        test_dataset = NPZDataset(test_data_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = VAE().to(device)
        model.load_state_dict(torch.load(model_load_path))
        print("Starting reconstruction testing...")
        test_reconstruction(model, test_loader)
        print("Reconstruction testing completed.")

    elif len(sys.argv) >= 5 and sys.argv[2] == "test_classifier":
        test_data_path = sys.argv[1]
        model_load_path = sys.argv[3]
        gmm_params_path = sys.argv[4]

        test_dataset = NPZDataset(test_data_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = VAE().to(device)
        model.load_state_dict(torch.load(model_load_path))
        print("Starting classification testing...")
        test_classifier(model, test_loader, gmm_params_path)
        print("Classification testing completed.")
