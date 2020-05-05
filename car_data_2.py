# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:41:33 2020

@author: Ernest Fedorowich
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#set the path on computer
car_data_path = (r'C:\Users\Ernest Fedorowich\Documents\projects\USA_cars_datasets.csv')
car_data = pd.read_csv(car_data_path)

#define the data set used, remove useless columns
df = car_data
df = df.drop(['country', 'condition'], axis=1)

#define the output variable (price)
y = car_data.price

#define the columns used for the model
feature_names = ['brand', 'model', 'year', 'mileage', 'title_status', 'state']

#defining the X output variable
X = car_data[feature_names]
auto_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# Break off the validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns to replace with one-hot encoding
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches (from www.kaggle.com)
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

print("MAE from One-Hot Encoding:") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
