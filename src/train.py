#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:09:35 2020

@author: pranavmandolkar
"""

import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
#TRAINING_DATA = /Users/pranavmandolkar/Public/ml/input/train_folds.csv
FOLD = int(os.environ.get("FOLD"))
#FOLD = 0
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
    }

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
   # df = pd.read_csv("/Users/pranavmandolkar/Public/ml/input/train_folds.csv")
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]
    
    y_train = train_df.target.values
    y_valid = valid_df.target.values
    
    train_df = train_df.drop(["id","target","kfold"], axis=1)
    valid_df = valid_df.drop(["id","target","kfold"], axis=1)
    
    valid_df = valid_df[train_df.columns]
    
    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + test_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
        
    
#    clf = ensemble.RandomForestClassifier(n_estimators= 200, n_jobs = -1, verbose=2)
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(y_valid, preds))
    
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
