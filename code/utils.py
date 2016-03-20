"""
__file__

	utils.py

__description__

	This file provides utils for data analysis

__author__

	Andrea Schioppa

"""

# !! Each module has its own global namespace
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import pandas as pd

# Create dummy vars in columnName
def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features

# Create Impacted variables of a factor
# exclude are columns excluded from this transformation
def ImpactData(train, test, exclude = []):

    features = train.columns[2:]
    print(type(features))
    for col in features:
        if((train[col].dtype == 'object') and (col not in exclude)):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
    return train, test

# For object variables fill out NAs
def NA_object_to_string(df, newname = "EMPTY"):
    for col in df.columns:
        if df[col].dtype == 'O' and sum(df[col].isnull())>0 :
            df.loc[df[col].isnull(), col] = newname # in the future try .fillna()


# convert object columns in df which are NOT in exclude to factors
# Note: factors are encode as integers, does NOT create dummies
def object_to_factor(df, exclude = []) :
    for col in df.columns:
        if col not in exclude and df[col].dtype == 'O':
            df[col], _ = pd.factorize(df[col]) 

