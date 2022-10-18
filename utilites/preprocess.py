#######################
### Library imports ###
#######################

# standard library
import os
import sys

# data packages
import numpy as np
import pandas as pd

# sklearn
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer


class Preprocessor(TransformerMixin):
    def __init__(
        self,
        seed = 2021
    ):
        self.seed = seed
        
    def fit(self, X, y = None, X_test = None):
        if X_test is not None:
            X = pd.concat([X, X_test], axis = 0, ignore_index = True)
    
        gene_feats = [col for col in X.columns if col.startswith('g-')]
        cell_feats = [col for col in X.columns if col.startswith('c-')]
        numeric_feats = gene_feats + cell_feats            
        categorical_feats = ['cp_time', 'cp_dose']        
        
        self._transformer = make_column_transformer(
            (OneHotEncoder(), categorical_feats),
            (StandardScaler(), numeric_feats)
        )
        self._transformer.fit(X)
        return self
        
    
    def transform(self, X):
        X_new = self._transformer.transform(X).astype("float32")
        return X_new
