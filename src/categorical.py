#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:54:18 2020

@author: pranavmandolkar
"""

"""
- label Encoding
- Binary Encoding
- One hot encoding
"""

from sklearn import preprocessing
import pandas as pd

class CategoricalFeatures:
    def __init__(
            self,
            df,
            categorical_features,
            encoding_type,
            handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names e.g ["ord_1", "ord_2", ....]
        handle_na: True/False
        """
        
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-999999")
                
        self.output_df = self.df.copy(deep=True)
        
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbb = preprocessing.LabelBinarizer()
            lbb.fit(self.df[c].values)
            val = lbb.transform(self.df[c].values) #array
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbb
            
        return self.output_df
            
    def _one_hot_encoding(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)
            
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot_encoding()
        else:
            raise Exception("Encoding type not understood")
   
            
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna("-999999")
                
            return dataframe
                
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            
            return dataframe
        
        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                dataframe[new_col_name] = val[:, j]
                
            return dataframe
                
        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
            
        else:
            raise Exception("Encoding type not understood")
            
                
            
if __name__ == "__main__":
    df = pd.read_csv("../input/train_categorical.csv")# .head(500)
    df_test = pd.read_csv("../input/test_categorical.csv")# .head(500)
    
    """
    # For Binarization
    train_idx = df["id"].values # To remember train data for cross validations
    test_idx = df_test["id"].values # To remember test data rows
    """
    
    # For OHE
    train_len = len(df)
    test_len = len(df_test)
    
    df_test["target"] = -1 # to add missing target column in df_test
    full_data = pd.concat([df, df_test]) # For handling error "y contains previously unseen labels: 'a885aacec'"
    
    
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats =  CategoricalFeatures(full_data, 
                                     categorical_features=cols,
                                     encoding_type="ohe",
                                     handle_na=True)
    full_data_transformed = cat_feats.fit_transform()

    """
    # For Binarization
    train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop=True)
    test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop=True)
    """
    # For OHE
    train_df = full_data_transformed[:train_len, :]
    test_df = full_data_transformed[train_len:, :]
    
    print(train_df.shape)
    print(test_df.shape)
    