#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:11:41 2020

@author: pranavmandolkar
"""
# Create Folds

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("/Users/pranavmandolkar/Public/ml/input/train.csv")
    df["kfold"] = -1
    
    df = df.sample(frac=1).reset_index(drop = True)
    
    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle = False, random_state = 42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv("input/train_folds.csv", index = False)

