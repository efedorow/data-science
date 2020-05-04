#took a data set from www.kaggle.com on US car sales (both used and new)
#created a model to predict the price of a car based on several parameters such as brand, mileage, year, etc
#in the end, a mean average error for price of 3951.57 was obtained for a set with an average price of 18767.67
#one-hot encoding had to be used since several inputs (e.g. brand, state) were stored as strings and had to be converted

#first import everything here
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#set the csv file path
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
auto_model = DecisionTreeRegressor(random_state=1)

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches (from www.kaggle.com)
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#print the output of mean average error
print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
