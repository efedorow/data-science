#tried to use lightgbm here but it failed because the data could not be encoded properly
#due to lightgbm's lack of support for multi-classes

import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

#get the data first
car_data_path = (r'C:\Users\Ernest Fedorowich\Documents\projects\USA_cars_datasets.csv')
car_data = pd.read_csv(car_data_path)


def get_data_splits(dataframe, valid_fraction=0.1):
    dataframe = dataframe.sort_values('price')
    valid_rows = int(len(dataframe)*valid_fraction)
    train = dataframe[:-valid_rows * 2]
    valid = dataframe[-valid_rows *2:-valid_rows]
    test = dataframe[-valid_rows:]
    
    return train, valid, test

def train_model(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(['price'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['price'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['price'])
    param = {'objective':'float'}
    num_round = 1000
    print("Training model!")
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval=False)
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(test['price'], valid_pred)
    print(f"Validations AUC score: {valid_score}")
    
    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['price'], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score
    
import category_encoders as ce
from category_encoders import CountEncoder
    
cat_features = ['brand', 'model', 'title_status', 'state', 'condition']
train, valid, test = get_data_splits(car_data)

count_enc = CountEncoder(cols=cat_features)
count_enc.fit(train[cat_features])

train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))
valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))

_ = train_model(train_encoded, valid_encoded)

print("Baseline model")
train, valid, test = get_data_splits(car_data)
bruh = train_model(train, valid)
