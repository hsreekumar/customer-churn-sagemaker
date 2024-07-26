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
from io import BytesIO
import logging

import subprocess
import sys
# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('shap')
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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
    def __init__(self, data, window_size):
        self.data = data.copy()
        # Scale numerical features
        numerical_features = [
            f'esent_rolling_mean_{window_size}', f'eopenrate_rolling_mean_{window_size}', f'eclickrate_rolling_mean_{window_size}', 
            f'avgorder_rolling_mean_{window_size}', f'ordfreq_rolling_mean_{window_size}', 
            f'esent_rolling_std_{window_size}', f'eopenrate_rolling_std_{window_size}', f'eclickrate_rolling_std_{window_size}', 
            f'avgorder_rolling_std_{window_size}', f'ordfreq_rolling_std_{window_size}'
        ]
        self.scaler = StandardScaler()
        self.data[numerical_features] = self.scaler.fit_transform(self.data[numerical_features])
        self.features = numerical_features + [col for col in self.data.columns if col.startswith('favday_') or col.startswith('city_')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx][self.features].values.astype(np.float32)
        target = self.data.loc[idx, 'retained']
        return torch.tensor(features), torch.tensor(target, dtype=torch.long)

# Function to download file from S3
def download_s3_file(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def train(args):
    window_size = args['window_size']
    
    # Extract bucket and key from S3 path
    train_data_path = args['train_data_path']
    test_data_path = args['test_data_path']
    
    train_bucket, train_key = train_data_path.replace("s3://", "").split("/", 1)
    test_bucket, test_key = test_data_path.replace("s3://", "").split("/", 1)
    
    # Download the files from S3
    logger.info(f"Downloading training data from s3://{train_bucket}/{train_key}")
    train_data = download_s3_file(train_bucket, train_key)
    logger.info(f"Downloading testing data from s3://{test_bucket}/{test_key}")
    val_data = download_s3_file(test_bucket, test_key)

    # Create CustomDataset instances
    train_dataset = CustomDataset(train_data, window_size)
    val_dataset = CustomDataset(val_data, window_size)

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Initialize model, optimizer, and criterion
    model = SimpleNN(input_size=len(train_dataset.features))
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info("Starting training loop")
    for epoch in range(args['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{args["epochs"]}, Batch {batch_idx}, Loss: {loss.item()}')

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
        logger.info(f'Epoch {epoch+1}/{args["epochs"]}, Validation AUC: {val_auc}')
        
        # Log validation AUC for SageMaker hyperparameter tuning
        logger.info(f'Validation AUC: {val_auc}')  # Ensure this format matches your metric definition

    # Save the model state_dict to the path /opt/ml/model (adjust path as needed)
    model_dir = '/opt/ml/model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)  # Save only the state_dict
    logger.info(f'Model saved to {model_path}')

    # Upload model to S3
    bucket_name = args['bucket_name']
    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket_name, f'model_{window_size}D/model.pth')
    logger.info(f'Model uploaded to s3://{bucket_name}/model_{window_size}D/model.pth')

    # Compute feature importance with SHAP
    logger.info("Computing feature importance with SHAP")
    background = train_dataset[:100][0]  # Use the first 100 samples from training data as background data
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(train_dataset[:][0])

    # Create a DataFrame for feature importances
    feature_importances = np.abs(shap_values[0]).mean(axis=0)  # Assuming binary classification, shap_values[0] for class 0
    feature_importances_df = pd.DataFrame({'feature': train_dataset.features, 'importance': feature_importances})
    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

    # Save feature importances to a CSV file
    feature_importances_path = os.path.join(model_dir, 'feature_importances.csv')
    feature_importances_df.to_csv(feature_importances_path, index=False)
    logger.info(f'Feature importances saved to {feature_importances_path}')


    # Upload feature importances to S3
    s3.upload_file(feature_importances_path, bucket_name, f'model_{window_size}D/feature_importances.csv')
    logger.info(f'Feature importances uploaded to s3://{bucket_name}/model_{window_size}D/feature_importances.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--bucket-name', type=str, required=True, metavar='B', help='the S3 bucket where models should be stored')
    parser.add_argument('--window-size', type=int, required=True, metavar='W', help='the rolling window size (e.g., 15, 30, 45)')
    parser.add_argument('--train-data-path', type=str, required=True, help='S3 path to the training data CSV file')
    parser.add_argument('--test-data-path', type=str, required=True, help='S3 path to the testing data CSV file')
    args = parser.parse_args()
    train(vars(args))
