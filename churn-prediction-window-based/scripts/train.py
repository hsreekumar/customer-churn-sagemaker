import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np
import argparse
import boto3

# Define your model class
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom dataset class for loading tabular data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data.copy()
        # Scale numerical features
        self.scaler = StandardScaler()
        self.data[['esent', 'eopenrate', 'eclickrate', 'avgorder', 'ordfreq']] = self.scaler.fit_transform(
            self.data[['esent', 'eopenrate', 'eclickrate', 'avgorder', 'ordfreq']]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx].drop('retained').values.astype(np.float32)
        target = self.data.loc[idx, 'retained']
        return torch.tensor(features), torch.tensor(target, dtype=torch.long)

def train(args):
    # Load preprocessed data from CSV files
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')
    val_data = pd.read_csv('/opt/ml/input/data/test/test.csv')

    # Create CustomDataset instances
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Initialize model, optimizer, and criterion
    model = SimpleNN(input_size=len(train_dataset.data.columns) - 1)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(args['epochs']):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{args["epochs"]}, Loss: {loss.item()}')

        # Validation step
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
        val_auc = roc_auc_score(all_targets, all_preds)
        print(f'Epoch {epoch+1}/{args["epochs"]}, Validation AUC: {val_auc}')
        model.train()

        # Log validation AUC for SageMaker hyperparameter tuning
        print(f'Validation AUC: {val_auc}')  # Ensure this format matches your metric definition

    # Save the model state_dict to the path /opt/ml/model (adjust path as needed)
    model_dir = '/opt/ml/model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)  # Save only the state_dict

    # Upload model to S3
    bucket_name = args['bucket_name']
    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket_name, 'model/model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--bucket-name', type=str, required=True, metavar='B', help='the S3 bucket where models should be stored')
    args = parser.parse_args()
    train(vars(args))
