import json
import boto3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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

def get_columns(window_size):
    if window_size == 15:
        return ['favday_Friday', 'favday_Monday', 'favday_Saturday', 'favday_Sunday', 
                'favday_Thursday', 'favday_Tuesday', 'favday_Wednesday', 'city_BLR', 
                'city_BOM', 'city_DEL', 'city_MAA', 'esent_rolling_mean_15', 
                'esent_rolling_std_15', 'eopenrate_rolling_mean_15', 'eopenrate_rolling_std_15', 
                'eclickrate_rolling_mean_15', 'eclickrate_rolling_std_15', 'avgorder_rolling_mean_15', 
                'avgorder_rolling_std_15', 'ordfreq_rolling_mean_15', 'ordfreq_rolling_std_15']
    elif window_size == 30:
        return ['favday_Friday', 'favday_Monday', 'favday_Saturday', 'favday_Sunday', 
                'favday_Thursday', 'favday_Tuesday', 'favday_Wednesday', 'city_BLR', 
                'city_BOM', 'city_DEL', 'city_MAA', 'esent_rolling_mean_30', 
                'esent_rolling_std_30', 'eopenrate_rolling_mean_30', 'eopenrate_rolling_std_30', 
                'eclickrate_rolling_mean_30', 'eclickrate_rolling_std_30', 'avgorder_rolling_mean_30', 
                'avgorder_rolling_std_30', 'ordfreq_rolling_mean_30', 'ordfreq_rolling_std_30']
    elif window_size == 45:
        return ['favday_Friday', 'favday_Monday', 'favday_Saturday', 'favday_Sunday', 
                'favday_Thursday', 'favday_Tuesday', 'favday_Wednesday', 'city_BLR', 
                'city_BOM', 'city_DEL', 'city_MAA', 'esent_rolling_mean_45', 
                'esent_rolling_std_45', 'eopenrate_rolling_mean_45', 'eopenrate_rolling_std_45', 
                'eclickrate_rolling_mean_45', 'eclickrate_rolling_std_45', 'avgorder_rolling_mean_45', 
                'avgorder_rolling_std_45', 'ordfreq_rolling_mean_45', 'ordfreq_rolling_std_45']
    else:
        raise ValueError("Unsupported window size")

def lambda_handler(event, context):
    try:
        logger.info("Lambda function started")

        # Parse request body
        body = json.loads(event['body'])
        data = body['data']
        bucket = body['bucket']
        window_size = body.get('window_size', 30)  # Default to 30 if not provided

        logger.info(f"Parsed body: {body}")
        logger.info(f"Bucket: {bucket}")
        logger.info(f"Window size: {window_size}")

        # Get columns based on window size
        columns = get_columns(window_size)
        test_data = pd.DataFrame(data, columns=columns)
        logger.info(f"Converted input data to DataFrame: {test_data}")

        # Define model paths based on window size
        model_key = f'model_{window_size}D/model.pth'

        # Scale the numerical features if necessary
        scaler_columns = [col for col in columns if 'rolling_mean' in col or 'rolling_std' in col]
        scaler = StandardScaler()
        test_data[scaler_columns] = scaler.fit_transform(test_data[scaler_columns])
        logger.info(f"Scaled test data: {test_data}")

        # Select features for the model
        features = test_data.values.astype(np.float32)
        logger.info(f"Features prepared for model: {features}")

        # Load the model from S3
        s3 = boto3.client('s3')
        model_path = f'/tmp/model_{window_size}D.pth'
        logger.info(f"Downloading model from S3 bucket {bucket} with key {model_key}")
        s3.download_file(bucket, model_key, model_path)

        # Instantiate the model with the correct input size
        model = SimpleNN(input_size=len(columns))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded and set to evaluation mode")

        # Convert features to tensor
        features_tensor = torch.tensor(features)
        logger.info(f"Features tensor: {features_tensor}")

        # Make predictions
        with torch.no_grad():
            outputs = model(features_tensor)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.numpy().tolist()
            logger.info(f"Predictions: {predictions}")

        return {
            'statusCode': 200,
            'body': json.dumps({'predictions': predictions})
        }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
