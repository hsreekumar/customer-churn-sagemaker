import pandas as pd
from sklearn.model_selection import train_test_split
import os


local_file_path = '/opt/ml/processing/input/data.csv'
print(f'Downloading from s3')
# Load data
data = pd.read_csv(local_file_path)
print(f'Read from s3')
# Convert date columns
data["firstorder"] = pd.to_datetime(data["firstorder"], errors='coerce')
data["lastorder"] = pd.to_datetime(data["lastorder"], errors='coerce')

# Drop rows with null values
data = data.dropna()

# Create new columns based on date differences
data["first_last_days_diff"] = (data['lastorder'] - data['firstorder']).dt.days
data['created'] = pd.to_datetime(data['created'])
data['created_first_days_diff'] = (data['created'] - data['firstorder']).dt.days

# Drop unnecessary columns
data.drop(['custid', 'created', 'firstorder', 'lastorder'], axis=1, inplace=True)

# Apply one-hot encoding to categorical columns
data = pd.get_dummies(data, prefix=['favday', 'city'], columns=['favday', 'city'])

# Split data into train and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Define directories for train and test data
train_dir = '/opt/ml/processing/train'
test_dir = '/opt/ml/processing/test'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save processed data to CSV files
train.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

# Log the shapes of the train and test datasets
print(f'Train data shape: {train.shape}')
print(f'Test data shape: {test.shape}')
