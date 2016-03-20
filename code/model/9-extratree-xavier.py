#########################################################################
# Trying ExtraTrees Classifiers
##########################################################################

import sys
import csv
import os
import time
import random

import pandas as pd
import numpy as np

# import configuration and utils
sys.path.append("../")
from param_config import config
from utils import * # so one does not have to append names

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics

# load raw data
train = pd.read_csv(''.join([config.data_dir, "train.csv"]))
test = pd.read_csv(''.join([config.data_dir, "test.csv"]))

# separate ID, target, drop some columns
train_id = train['ID'].values # gets an nparray
train_target = train['target'].values
FeaturesToDrop = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63',
                    'v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109',
                    'v110','v116','v117','v118','v119','v123','v124','v128']
train = train.drop(['ID','target'] + FeaturesToDrop,axis=1)
test_id = test['ID'].values
test = test.drop(['ID'] + FeaturesToDrop,axis=1)

# Create Factors and Impute missing values trivially
# The column type of factors is int64
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name]) 
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999


############################
# Save a copy of this data #
############################
# remerge train
Alldf = pd.concat([pd.DataFrame({'ID' : train_id, 'target' : train_target}),
                   train], axis = 1)
# append test and sort by ID
Alldf = Alldf.append(pd.concat([pd.DataFrame({'ID' : test_id, 'target' : None}),
                                test], axis = 1))
Alldf = Alldf.sort('ID')
Alldf.to_csv(''.join([config.feat_dir, 'kaggle-C1-16-3-16.csv']), index = False)

######################
# Train an ExtraTree #
######################
X_train = train
X_test = test

######################
# 18.027162166436515 #
######################
t0 = time.time()
extc = ExtraTreesClassifier(n_estimators=850, max_features= 60, criterion= 'entropy', min_samples_split= 4,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1) # -1 = number of jobs = cores   

extc.fit(X_train, train_target)
t0 = time.time() - t0

# predict
y_pred = extc.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": test_id, "PredictedProb": y_pred[:,1]}).to_csv(''.join([config.subm_dir,
                                                                            'sub-C1-16-3-16.csv']),
                                                                   index=False)
##########################
# Add impacted variables #
##########################
# Reread data
train = pd.read_csv(''.join([config.data_dir, "train.csv"]))
test = pd.read_csv(''.join([config.data_dir, "test.csv"]))

# Variables to impact
to_impact = [col for col in train.columns if train[col].dtype == 'object']
train_to_impact = train[['target'] + to_impact]
test_to_impact = test[to_impact]

# Impact: issue with loading the script; why nb is a local name?
train_impacted, test_impacted = ImpactData(train_to_impact,
                                           test_to_impact, exclude = ['v22'])
train_impacted = train_impacted.drop(['target', 'v3', 'v22'], axis = 1)
test_impacted = test_impacted.drop(['v3', 'v22'], axis = 1)

train_impacted = train_impacted.rename(columns = dict(zip(train_impacted.columns.values,
                                         map(lambda x: 'Imp_' + x, train_impacted.columns.values))))
test_impacted = test_impacted.rename(columns = dict(zip(test_impacted.columns.values,
                                         map(lambda x: 'Imp_' + x, test_impacted.columns.values))))

# merge in train, test
train = pd.concat([train, train_impacted], axis = 1)
test = pd.concat([test, test_impacted], axis = 1)

# separate ID, target, drop some columns
train_id = train['ID'].values # gets an nparray
train_target = train['target'].values
train = train.drop(['ID', 'target'], axis = 1)
test_id = test['ID'].values
test = test.drop(['ID'],axis=1)

# Create Factors and Impute missing values trivially
# The column type of factors is int64
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name]) 
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999


############################
# Save a copy of this data #
############################
# remerge train
Alldf = pd.concat([pd.DataFrame({'ID' : train_id, 'target' : train_target}),
                   train], axis = 1)
# append test and sort by ID
Alldf = Alldf.append(pd.concat([pd.DataFrame({'ID' : test_id, 'target' : None}),
                                test], axis = 1))
Alldf = Alldf.sort('ID')
Alldf.to_csv(''.join([config.feat_dir, 'kaggle-C2-16-3-16.csv']), index = False)

######################
# Train an ExtraTree #
######################
X_train = train
X_test = test

######################
# 17.09431314865748  #
######################
t0 = time.time()
extc = ExtraTreesClassifier(n_estimators=850, max_features= 60, criterion= 'entropy', min_samples_split= 4,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1) # -1 = number of jobs = cores   

extc.fit(X_train, train_target)
t0 = time.time() - t0

# predict
y_pred = extc.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": test_id, "PredictedProb": y_pred[:,1]}).to_csv(''.join([config.subm_dir,
                                                                            'sub-C2-16-3-16.csv']),
                                                                   index=False)

##########################################################################
# Test on imputed-B* data and kaggle-C2
# This time a Reps = 3, K = 3 CV
##########################################################################
# read
Alldf = pd.read_csv(''.join([config.feat_dir, 'imputed-B1-16-3-4.csv']))

# fill NAs in factor variables
NA_object_to_string(Alldf)
# convert object columns to factors = int64
object_to_factor(Alldf)

# Train/Test split
X_train = Alldf.loc[Alldf['target'].notnull(),:]
X_test = Alldf.loc[Alldf['target'].isnull(),:]

ID_train = X_train['ID'].values
y_train = X_train['target'].values
ID_test = X_test['ID'].values

X_train = X_train.drop(['ID', 'target'], axis = 1)
X_test = X_test.drop(['ID', 'target'], axis = 1)


scores = [0, 0, 0]
kf = KFold(n = X_train.shape[0], n_folds = 3, random_state = 15)
extc = ExtraTreesClassifier(n_estimators=900, max_features= 80, criterion= 'entropy', min_samples_split= 4,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1, random_state = 2017) 

################################################################
# 53.765311082204185
# CV_mean : 0.45701511482758911, CV_sd : 0.0010578112043377035
################################################################
t0 = time.time()
cv_iter = 1
for train_index, cv_index in kf:
    print 'CV iteration', cv_iter
    XX_train, CV_train = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # must use iloc
    yy_train, ycv_true = y_train[train_index], y_train[cv_index]

    
    #print XX_train.head(11)
    #print yy_train[0:10]
    extc.fit(XX_train, yy_train)
    ycv_pred = extc.predict_proba(CV_train)

    #print metrics.log_loss(ycv_true, ycv_pred)
    scores[cv_iter-1] = metrics.log_loss(ycv_true, ycv_pred)
    cv_iter = cv_iter + 1
t0 = time.time() - t0

###################################
# Redo on "kaggle-C2-16-3-16.csv" #
###################################

# read
Alldf = pd.read_csv(''.join([config.feat_dir, 'kaggle-C2-16-3-16.csv']))

# fill NAs in factor variables
NA_object_to_string(Alldf)
# convert object columns to factors = int64
object_to_factor(Alldf)

# Train/Test split
X_train = Alldf.loc[Alldf['target'].notnull(),:]
X_test = Alldf.loc[Alldf['target'].isnull(),:]

ID_train = X_train['ID'].values
y_train = X_train['target'].values
ID_test = X_test['ID'].values

X_train = X_train.drop(['ID', 'target'], axis = 1)
X_test = X_test.drop(['ID', 'target'], axis = 1)


scores = [0, 0, 0]
kf = KFold(n = X_train.shape[0], n_folds = 3, random_state = 15)
extc = ExtraTreesClassifier(n_estimators=900, max_features= 80, criterion= 'entropy', min_samples_split= 4,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1, random_state = 2017) 

################################################################
# 60.98842646678289
# CV_mean : 0.45988224449330434, CV_sd : 0.0019215082746658414]
################################################################
t0 = time.time()
cv_iter = 1
for train_index, cv_index in kf:
    print 'CV iteration', cv_iter
    XX_train, CV_train = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # must use iloc
    yy_train, ycv_true = y_train[train_index], y_train[cv_index]

    
    #print XX_train.head(11)
    #print yy_train[0:10]
    extc.fit(XX_train, yy_train)
    ycv_pred = extc.predict_proba(CV_train)

    #print metrics.log_loss(ycv_true, ycv_pred)
    scores[cv_iter-1] = metrics.log_loss(ycv_true, ycv_pred)
    cv_iter = cv_iter + 1
t0 = time.time() - t0

###################################
# Redo on "imputed-B2-16-3-4.csv" #
###################################

# read
Alldf = pd.read_csv(''.join([config.feat_dir, 'imputed-B2-16-3-4.csv']))

# fill NAs in factor variables
NA_object_to_string(Alldf)
# convert object columns to factors = int64
object_to_factor(Alldf)

# Train/Test split
X_train = Alldf.loc[Alldf['target'].notnull(),:]
X_test = Alldf.loc[Alldf['target'].isnull(),:]

ID_train = X_train['ID'].values
y_train = X_train['target'].values
ID_test = X_test['ID'].values

X_train = X_train.drop(['ID', 'target'], axis = 1)
X_test = X_test.drop(['ID', 'target'], axis = 1)


scores = [0, 0, 0]
kf = KFold(n = X_train.shape[0], n_folds = 3, random_state = 15)
extc = ExtraTreesClassifier(n_estimators=900, max_features= 80, criterion= 'entropy', min_samples_split= 4,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1, random_state = 2017) 

################################################################
# 48.972529248396555
# CV_mean : 0.45962881844869496, CV_sd : 0.0015868649025005041
################################################################
t0 = time.time()
cv_iter = 1
for train_index, cv_index in kf:
    print 'CV iteration', cv_iter
    XX_train, CV_train = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # must use iloc
    yy_train, ycv_true = y_train[train_index], y_train[cv_index]

    
    #print XX_train.head(11)
    #print yy_train[0:10]
    extc.fit(XX_train, yy_train)
    ycv_pred = extc.predict_proba(CV_train)

    #print metrics.log_loss(ycv_true, ycv_pred)
    scores[cv_iter-1] = metrics.log_loss(ycv_true, ycv_pred)
    cv_iter = cv_iter + 1
t0 = time.time() - t0


###################################
# Redo on "imputed-B3-16-3-6.csv" #
###################################

# read
Alldf = pd.read_csv(''.join([config.feat_dir, 'imputed-B3-16-3-6.csv']))
Alldf = Alldf.drop(['Ts_v23'], axis = 1) # Error, all column is read as NaN; probably pandas error
# The variable was not much significant, should not make much difference.
# fill NAs in factor variables
NA_object_to_string(Alldf)
# convert object columns to factors = int64
object_to_factor(Alldf)

# Train/Test split
X_train = Alldf.loc[Alldf['target'].notnull(),:]
X_test = Alldf.loc[Alldf['target'].isnull(),:]

ID_train = X_train['ID'].values
y_train = X_train['target'].values
ID_test = X_test['ID'].values

X_train = X_train.drop(['ID', 'target'], axis = 1)
X_test = X_test.drop(['ID', 'target'], axis = 1)


scores = [0, 0, 0]
kf = KFold(n = X_train.shape[0], n_folds = 3, random_state = 15)
extc = ExtraTreesClassifier(n_estimators=900, max_features= 80, criterion= 'entropy', min_samples_split= 4,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1, random_state = 2017) 

################################################################
# 83.44245056708654
# CV_mean : 0.45988467004985173, CV_sd : 0.0014078062441815961
################################################################
t0 = time.time()
cv_iter = 1
for train_index, cv_index in kf:
    print 'CV iteration', cv_iter
    XX_train, CV_train = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # must use iloc
    yy_train, ycv_true = y_train[train_index], y_train[cv_index]

    
    #print XX_train.head(11)
    #print yy_train[0:10]
    extc.fit(XX_train, yy_train)
    ycv_pred = extc.predict_proba(CV_train)

    #print metrics.log_loss(ycv_true, ycv_pred)
    scores[cv_iter-1] = metrics.log_loss(ycv_true, ycv_pred)
    cv_iter = cv_iter + 1
t0 = time.time() - t0

# Fitting models from the three files

file_list = ['imputed-B1-16-3-4.csv', 'imputed-B2-16-3-4.csv', 'imputed-B3-16-3-6.csv']

for i, fname in enumerate(file_list):
    print 'Processing', i+1, ':', fname
    # read and preprocess
    Alldf = pd.read_csv(''.join([config.feat_dir, fname]))
    if fname == 'imputed-B3-16-3-6.csv':
        Alldf = Alldf.drop(['Ts_v23'], axis = 1)
        
    NA_object_to_string(Alldf)
    # convert object columns to factors = int64
    object_to_factor(Alldf)

    # Train/Test split
    X_train = Alldf.loc[Alldf['target'].notnull(),:]
    X_test = Alldf.loc[Alldf['target'].isnull(),:]

    ID_train = X_train['ID'].values
    y_train = X_train['target'].values
    ID_test = X_test['ID'].values

    X_train = X_train.drop(['ID', 'target'], axis = 1)
    X_test = X_test.drop(['ID', 'target'], axis = 1)

    # fit
    extc = ExtraTreesClassifier(n_estimators=900, max_features= 80, criterion= 'entropy',
                                min_samples_split= 4, max_depth= 40, min_samples_leaf= 2,
                                n_jobs = 2, random_state = 2017) 
    extc.fit(X_train, y_train)
    
    # predict
    y_pred = extc.predict_proba(X_test)

    # write subm
    sub_name = ''.join(['sub-C', str(i+3), '-16-3-18.csv'])
    sub_name = ''.join([config.subm_dir, sub_name])
    pd.DataFrame({"ID": ID_test, "PredictedProb": y_pred[:,1]}).to_csv(sub_name, index=False)
