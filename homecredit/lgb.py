# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

os.chdir(r"C:\Users\JG\Documents\GitHub\learning-journey\homecredit")
# print(os.listdir("../input"))
print(os.listdir("input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

# Write data into dataframes
app_train = pd.read_csv('input/application_train.csv')
app_test = pd.read_csv('input/application_test.csv')

# Use label encoder for categorical columns with 2 or less
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_count = 0

cat_indices = []

for i, col in enumerate(app_train):
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
        
            # Track how many columns were label encoded
            le_count += 1
            cat_indices.append(i)
print("%d columns were label encoded." % le_count)
print(cat_indices)

# One-hot encoding for all categorical features
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print("Fully encoded training data shape: ", app_train.shape)
print("Fully encoded test data shape: ", app_test.shape)

train_labels = app_train['TARGET']

# Align the data to only include columns present in both training and test data sets
app_train, app_test = app_train.align(app_test, join = 'inner', axis=1)

app_train['TARGET'] = train_labels

from sklearn.preprocessing import MinMaxScaler, Imputer

if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()


# Make a list of all the features
features = list(train.columns)

test = app_test.copy()

imputer = Imputer(strategy = 'median')

scaler = MinMaxScaler(feature_range=(0,1))

# Fit and train imputer
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

# Repeat with scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape:', train.shape)
print('Test data shape:', test.shape)

# Preparing dataframes for input into model
columns = app_test.columns.values

train_df = pd.DataFrame(train, columns=columns)
test_df = pd.DataFrame(test, columns=columns)

train_df['TARGET'] = train_labels


print('Train dataframe shape:', train_df.shape)
print('Test dataframe shape:', test_df.shape)

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import gc

def model(features, test_features, n_folds = 5):
    '''
    features (pd.DataFrame):
        training dataframe, must include TARGET column
    test_features (pd.DataFrame):
        test dataframe
    n_folds (int):
        default of 5
    
    '''
    print('Training data shape:', features.shape)
    print('Test data shape:', test_features.shape)
    
    # Extract IDs and labels
    train_ids = features['SK_ID_CURR']
    test_ids = app_test['SK_ID_CURR']    
    labels = features['TARGET']
    
    # Drop ID and target columns
    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])
    

    feature_names = features.columns
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)  
    
    # Create the StratifiedKFold object
    kfold = KFold(n_splits = n_folds, shuffle=True, random_state = 50)
    
    # Empty array for feature importances
    feature_importances = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold predictions
    fold_predictions = np.zeros(features.shape[0])
    
    # Empty list for train and valid scores
    train_scores = []
    valid_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in kfold.split(features, labels):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', metric = 'binary_logloss',
                                   is_unbalance=True, learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.8, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50, max_depth = 5, num_leaves = 20, feature_fraction = 0.5)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importances += model.feature_importances_ / kfold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / kfold.n_splits
        
        # Record the out of fold predictions
        fold_predictions[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    
    # Make submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make feature importance dataframe
    feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, fold_predictions)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances_df, metrics  
    
    

submission, fi, metrics = model(train_df, test_df, n_folds=4)
print('Baseline metrics')
print(metrics)