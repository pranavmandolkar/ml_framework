#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:14:31 2020

@author: pranavmandolkar
"""

from sklearn import ensemble

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators= 200, n_jobs = -1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators= 200, n_jobs = -1, verbose=2)
    
    }

