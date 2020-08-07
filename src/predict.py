#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:12:16 2020

@author: pranavmandolkar
"""

import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import joblib

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
#TRAINING_DATA = /Users/pranavmandolkar/Public/ml/input/train_folds.csv
MODEL = os.environ.get("MODEL")


def predict():
   df = pd.read_csv(TEST_DATA)
   # df = pd.read_csv("/Users/pranavmandolkar/Public/ml/input/train_folds.csv")
   test_idx = df["id"].values
   predictions = None
   
   for FOLD in range(5):
       print(FOLD)
       df = pd.read_csv(TEST_DATA)
       encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
       cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
       for c in encoders:
           print(c)
           lbl = encoders[c]
           df.loc[:, c] = lbl.transform(df[c].values.tolist())
            
    #    clf = ensemble.RandomForestClassifier(n_estimators= 200, n_jobs = -1, verbose=2)
       clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
       
      
       df = df[cols]
       preds = clf.predict_proba(df)[:, 1]
       
       if FOLD == 0:
           predictions = preds 
       else:
           predictions += preds
           
           
           
           
   predictions /= 5
    
   sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
   return sub
       

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)
    