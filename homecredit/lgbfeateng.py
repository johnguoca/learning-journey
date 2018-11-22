# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import lightgbm as lgb
from contextlib import contextmanager
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.metrics import roc_auc_score


os.chdir(r"C:\Users\JG\Documents\GitHub\learning-journey\homecredit")
# print(os.listdir("../input"))
print(os.listdir("input"))

# Will print time used for each function
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} was done in {:0f}s".format(title, time.time()-t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Pre-process application_test and application_train
def application_train_test(num_rows = None, nan_as_category = False):
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
    
    app_train['DAYS_EMPLOYED_PERC'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
    app_train['ANNUITY_INCOME_PERC'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    app_train['PAYMENT_RATE'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
    
    df = app_train.append(app_test).
    _index()
    
    del app_test
    
    gc.collect()
    
    return df

def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('input/bureau.csv')
    bb = pd.read_csv('input/bureau_balance.csv')
    
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    
    # These columns are one-hot encoded, so mean can be interpreted as a "count" for that category
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    
    # Give a new name to these columns
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    bureau = bureau.join(bb_agg, on='SK_ID_BUREAU')    
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    # The ** syntax here essentially combines the two dictionaries
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    gc.collect()
    
    return bureau_agg

# Pre-process previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category)
    
    # Replace the 365243 error
    columns_with_error = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in columns_with_error: prev[col].replace(365243, np.nan, inplace=True)
    
    # Add possibly useful features
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['DOWN_CREDIT_PERC'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    gc.collect()
    return prev_agg


    

# =============================================================================
# from sklearn.preprocessing import MinMaxScaler, Imputer
# 
# if 'TARGET' in app_train:
#     train = app_train.drop(columns = ['TARGET'])
# else:
#     train = app_train.copy()
# 
# 
# # Make a list of all the features
# features = list(train.columns)
# 
# test = app_test.copy()
# 
# imputer = Imputer(strategy = 'median')
# 
# scaler = MinMaxScaler(feature_range=(0,1))
# 
# # Fit and train imputer
# imputer.fit(train)
# train = imputer.transform(train)
# test = imputer.transform(test)
# 
# # Repeat with scaler
# scaler.fit(train)
# train = scaler.transform(train)
# test = scaler.transform(test)
# 
# print('Training data shape:', train.shape)
# print('Test data shape:', test.shape)
# 
# # Preparing dataframes for input into model
# columns = app_test.columns.values
# 
# train_df = pd.DataFrame(train, columns=columns)
# test_df = pd.DataFrame(test, columns=columns)
# 
# train_df['TARGET'] = train_labels
# 
# 
# print('Train dataframe shape:', train_df.shape)
# print('Test dataframe shape:', test_df.shape)
# =============================================================================


def kfold_lightgbm(df, num_folds = 5, stratified = False, debug=False):
    '''
    df (pd.DataFrame):
        Dataframe of training and test appended
    num_folds (int):
        default of 5
    
    '''
    # Split training and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    print('Training data shape:', train_df.shape)
    print('Test data shape:', test_df.shape)
    
    del df
    gc.collect()
    
    # Extract IDs and labels
    train_ids = train_df['SK_ID_CURR']
    test_ids = test_df['SK_ID_CURR']    
    labels = train_df['TARGET']
    
    # Drop columns that are not features
    cols_to_drop = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']
    for col in cols_to_drop:
        if col in train_df: train_df = train_df.drop(columns = col)
        if col in test_df: test_df = test_df.drop(columns = col)
    

    feature_names = train_df.columns
    
    # Convert to np arrays
    features = np.array(train_df)
    test_features = np.array(test_df)  
    
    # Create the fold object
    if stratified:
        fold = StratifiedKFold(n_splits = num_folds, shuttle=True, random_state = 101)
    else:
        fold = KFold(n_splits = num_folds, shuffle=True, random_state = 101)
    
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
    for train_indices, valid_indices in fold.split(features, labels):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importances += model.feature_importances_ / fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / fold.n_splits
        
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
    
def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Run LightGBM with kfold"):
        submission, fi, metrics = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)
        print('Baseline metrics')
        print(metrics)
        submission.to_csv(submission_file_name, index=False)
if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()    


