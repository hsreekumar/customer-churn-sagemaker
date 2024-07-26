import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def add_rolling_features(data, window_size):
    for column in ['esent', 'eopenrate', 'eclickrate', 'avgorder', 'ordfreq']:
        data[f'{column}_rolling_mean_{window_size}'] = data[column].rolling(window=window_size).mean()
        data[f'{column}_rolling_std_{window_size}'] = data[column].rolling(window=window_size).std()
    data.fillna(0, inplace=True)
    return data

data = pd.read_csv('/opt/ml/processing/input/data.csv')
data["firstorder"] = pd.to_datetime(data["firstorder"], errors='coerce')
data["lastorder"] = pd.to_datetime(data["lastorder"], errors='coerce')
data['created'] = pd.to_datetime(data['created'])

data = data.dropna()

data["first_last_days_diff"] = (data['lastorder'] - data['firstorder']).dt.days
data['created_first_days_diff'] = (data['created'] - data['firstorder']).dt.days

data.drop(['custid', 'created', 'firstorder', 'lastorder'], axis=1, inplace=True)
data = pd.get_dummies(data, prefix=['favday', 'city'], columns=['favday', 'city'])

for window_size in [15, 30, 45]:
    window_data = add_rolling_features(data.copy(), window_size)
    train, test = train_test_split(window_data, test_size=0.2, random_state=42)

    window_train_dir = f'/opt/ml/processing/train_{window_size}D'
    window_test_dir = f'/opt/ml/processing/test_{window_size}D'
    os.makedirs(window_train_dir, exist_ok=True)
    os.makedirs(window_test_dir, exist_ok=True)

    train.to_csv(os.path.join(window_train_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(window_test_dir, 'test.csv'), index=False)
