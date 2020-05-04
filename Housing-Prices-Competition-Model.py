#for this script, I utilized an approach where I dropped columns without data rather than using 
#the less effective imputation approach
#I got a score of 17978.52063
#the goal is to create a model to estimate the final price of a home based on 
#many factors such as the number of bathrooms or lot size, etc

#import stuff used like the csv files
import pandas as pd
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
    
#import the model used
from sklearn.model_selection import train_test_split

#here we define the X input variables
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

#remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

#we also have to define an output variable in terms of the home price
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

#here we break off the validation set from the training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
 
#this line removes an entire column if there is any missing data
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

#the data set is modified to get rid of columns with missing data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1) 
final_X_train = X_train.drop(cols_with_missing, axis=1)
final_X_valid = X_valid.drop(cols_with_missing, axis=1)

#use the random forest model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

#apply this concept to the real competition data
cols_with_missing = [col for col in X_test.columns if X_test[col].isnull().any()]
final_X_test = X_test.drop(cols_with_missing, axis=1)
preds_test = model.predict(final_X_test)

#finally save the test predictions to file for submission
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
