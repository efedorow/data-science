#here I tried to create a tensorflow ML model for the kaggle Titanic machine learning competition
#however it failed to work due to an issue with the dimensions of the model input
#I tried to fix this but did not have the technical knowledge or stackoverflow guidance
#otherwise the code probably would've worked fine

#the idea is to use a variety of given passenger data (e.g. age, fare price, etc) to predict
#whether or not they survived the sinking of the titanic

#import some of the modules used here
import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import matplotlib as plt

#locate the files on pc
titanic_train_path = (r'C:\Users\Ernest Fedorowich\Documents\projects\titanic-train.csv')
titanic_train = pd.read_csv(titanic_train_path)
train = titanic_train
titanic_test_path = (r'C:\Users\Ernest Fedorowich\Documents\projects\titanic-test.csv')
titanic_test = pd.read_csv(titanic_test_path)
test = titanic_test

#create a function to clean the data set
#fill in missing data for Fare and Age
#convert strings for Sex and Embarking location to numerical values rather than use One-Hot encoding
def data_cleaner(data):  
    #fill in the unavailable fare values with the median value
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    
    #fill in the unavailable age values with the median value
    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
    
    #convert the strings for sex to corresponding values
    data.loc[data['Sex']=='male', 'Sex'] = 0
    data.loc[data['Sex']=='female', 'Sex'] = 1
    
    #convert the value for embarking location to corresponding values
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data['Embarked']=='S', 'Embarked'] = 0
    data.loc[data['Embarked']=='C', 'Embarked'] = 1
    data.loc[data['Embarked']=='Q', 'Embarked'] = 2

#apply the cleaning function
data_cleaner(train)
data_cleaner(test)

#get rid of useless data such as cabin or name
#could analyze the data to see how much cabin data was missing, but it seemed like a fair amount based on visual judgement
drop_column = ['Cabin', 'Name']
train.drop(drop_column, axis=1, inplace=True)
test.drop(drop_column, axis=1, inplace=True)

all_data = [train, test]    

#create a new column for the family size (single people were less likely to make it onto a life raft)
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#create a new column for the value of passengers' fares (higher fare meant better survival chance)
for dataset in all_data:
    dataset['Fare_Range'] = pd.cut(dataset['Fare'], bins=[0, 7.91, 14.45, 120],
                                   labels=['Low_Fare', 'Median_Fare', 'High_Fare'])

#make a copy of the dataset for the training to avoid loss
traindf = train
testdf = test

all_data_2 = [traindf, testdf]

#drop these values
for dataset in all_data_2:
    drop_column = ['Age', 'Fare', 'Ticket']
    dataset.drop(drop_column, axis=1, inplace=True)
    
#remove the passenger id from training data
drop_column = ['PassengerId']
traindf.drop(drop_column, axis=1, inplace=True)

#call the copies created to avoid data loss
traindf = pd.get_dummies(traindf, columns = ['Sex', 'Embarked', 'Fare_Range'],
                         prefix=['Sex', 'Emb_Type', 'Fare_Type'])

#repeat for the testing data
testdf = pd.get_dummies(testdf, columns = ['Sex', 'Embarked', 'Fare_Range'],
                         prefix=['Sex', 'Emb_Type', 'Fare_Type'])


#create a correlation plot to show correlation between the different features
#eg how is survival correlated to fare price
#this is a handy bit of feature engineering I learned
#sns.heatmap(traindf.corr(),annot=True,linewidths=0.2)

#the target is the passenger's survival
target = traindf['Survived'].values
features = traindf[['Pclass', 'Sex_0', 'Sex_1', 'Emb_Type_0', 'Emb_Type_1', 'Emb_Type_2',
                    'Fare_Type_Low_Fare', 'Fare_Type_Median_Fare', 'Fare_Type_High_Fare']].values

column_names = ['Pclass', 'Sex_0', 'Sex_1', 'Emb_Type_0', 'Emb_Type_1', 'Emb_Type_2',
                    'Fare_Type_Low_Fare', 'Fare_Type_Median_Fare', 'Fare_Type_High_Fare']

feature_names = column_names[:-1]
label_name = column_names[-1]
                    
batch_size = 32

#set the training data for the tensorflow model
train = tf.data.experimental.make_csv_dataset(
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)                    

#repackage the features dictionary into a single array
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)                 
                    
train = train.map(pack_features_vector)
                
features, labels = next(iter(train))  

  
import tensorflow as tf
import keras        

#here we use tensorflow for our training data
model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(9,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
        ])

#now we model our predictions on the test data
predictions = model(testdf)

tf.nn.softmax(predictions[:5])

#find the loss between the labels and predictions
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#create a function to find the loss for the model based on training data
def loss_model(model, x, y, training):
    y_ = model(x, training=traindf)
    return loss_object(y_true=y, y_pred=y_)

#print the values from the loss test above
loss_final = loss_model(model, features, target, training=False)
print("Loss test: {}".format(loss_final))

#create a function to find the gradients used for optimization
def gradient_func(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_model(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#optimize the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

#print the loss values for both steps
loss_value, grads = gradient_func(model, features,labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
            
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(), 
                                  loss_model(model, features, labels, training=True).numpy()))

#from the tensorflow tutorial guide

num_epochs = 201

#we must iterate each epoch
#and make a prediction and compare it to the label, find loss and gradients
#use optimizer to update model variables

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train:
    # Optimize the model
    loss_value, grads = gradient_func(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
                                                                
test = tf.data.experimental.test_data.csv(
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

#apply the pack_features_vector from above
test = test.map(pack_features_vector)

#find the accuracy of the test
test_accuracy = tf.keras.metrics.Accuracy()

#evaluate the model based on each example
for (x, y) in test:
    #use training=False here because we dropped some columns earlier
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
    
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
  
#output the predictions based on test data to a csv file
def write_prediction(prediction, name):
    PassengerId = np.array(test['PassengerId']).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ['Survived'])
    solution.to_csv(name, index_label = ['PassengerId'])
