import subprocess
import sys
import tarfile
import os

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('torch')
install('torchvision')
install('numpy')
install('smdebug')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# Define your model class (make sure this matches the model definition used during training)
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

def main():
    # Load test data
    test_data = pd.read_csv('/opt/ml/processing/test/test.csv')

    # Preprocess test data
    scaler = StandardScaler()
    test_data[['esent', 'eopenrate', 'eclickrate', 'avgorder', 'ordfreq']] = scaler.fit_transform(
        test_data[['esent', 'eopenrate', 'eclickrate', 'avgorder', 'ordfreq']]
    )

    features = test_data.drop('retained', axis=1).values.astype(np.float32)
    targets = test_data['retained'].values

    print(f"features shape: {features.shape[1]}")
    
    # Extract the model from the tar.gz file
    model_tar_path = '/opt/ml/processing/models/model.tar.gz'
    extract_path = '/opt/ml/processing/models/'
    with tarfile.open(model_tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

    # Assuming the extracted model file is named 'model.pth'
    model_file_path = os.path.join(extract_path, 'model.pth')
    
    # Instantiate the model
    model = SimpleNN(input_size=features.shape[1])
    print("Downloading model state")
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded and set to evaluation mode")

    # Convert features to tensor
    features_tensor = torch.tensor(features)

    # Make predictions
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)

    # Calculate accuracy (example metric)
    accuracy = (predicted.numpy() == targets).mean()

    fpr, tpr, thresholds = roc_curve(targets, predicted.numpy())
    auc_score = auc(fpr, tpr)
    report_dict = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }

    # Save evaluation results
    evaluation_results = {
        'metric': 'accuracy',
        'value': accuracy.item()  # Ensure accuracy is converted to a float
    }

    with open('/opt/ml/processing/evaluation/evaluation.json', 'w') as f:
        json.dump(report_dict, f)

if __name__ == '__main__':
    main()
