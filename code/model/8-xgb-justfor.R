##########################################################################
# based on https://www.kaggle.com/justfor/bnp-paribas-cardif-claims-management/xgb-cross-val-and-feat-select
# XGboost based on Boruta feature selection
##########################################################################

library(bit64)
library(data.table) # more efficient
library(xgboost)
library(caret)
library(ggplot2) # already loaded by caret
library(dplyr)
library(Boruta)
library(mlr)

setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code/model')
source('../param_config.R') 
source('../utils.R')
source('metrics.R')

# Use features with Random Imputation
Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-B2-16-3-4.csv", sep = ''))

# Boruta data
load(paste(ParamConfig$output_dir, 'boruta-16-3-8.RData', sep = ''))
boruta.smry <- arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)), desc(medianImp))

# Remove features suggested by Boruta
boruta.to.remove <- c('v72', 'v62', 'v112', 'v107',
                      'v125', 'v75', 'v71', 'v91',
                      'v74', 'v52', 'v22', 'v3')

# remove other less significant variables
further.to.remove <- grep('(Nr|Imp|T\\w+|Num|RFreq|FFreq|It|Len)', boruta.smry$attr[206:258], value = TRUE)

highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128")

boruta.to.remove <- unique(c(boruta.to.remove, further.to.remove, highCorrRemovals))
Alldf[, c(boruta.to.remove) := NULL]

# transform to factors
num.to.fac <- intersect(c('v38', 'v62', 'v72', 'v129'), names(Alldf))
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)

# generic parameters
param0 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic"
        , "eval_metric" = "logloss"
        , "eta" = 0.05
        , "subsample" = 0.9
        , "colsample_bytree" = 0.8
        , "min_child_weight" = 1
        , "max_depth" = 10
        )

train <- data.matrix(Alldf[!is.na(target), -c(1,2), with = FALSE])
train.label <- Alldf[!is.na(target), target]
test <- data.matrix(Alldf[is.na(target), -c(1,2), with = FALSE])

xgtrain = xgb.DMatrix(train, label = train.label, missing=NA)
xgtest = xgb.DMatrix(test, missing=NA)

# Do cross-validation with xgboost - xgb.cv
XGB_docv <- function(param0, iter) {
    model_cv = xgb.cv(
            params = param0
            , nrounds = iter
            , nfold = 3
            , data = xgtrain
            , early.stop.round = 10
            , maximize = FALSE
            , nthread = 8
            )
    gc()
    best <- min(model_cv$test.logloss.mean)
    bestIter <- which(model_cv$test.logloss.mean == best)
    
    cat("\n",best, bestIter,"\n")
    print(model_cv[bestIter])
    
    bestIter-1
}

XGB_doTest <- function(param0, iter) {
    watchlist <- list('train' = xgtrain)
    model = xgb.train(
            nrounds = iter
            , params = param0
            , data = xgtrain
            , watchlist = watchlist
            , print.every.n = 20
            , nthread = 8
            )
    p <- predict(model, xgtest)
    rm(model)
    gc()
    p
}

##############################
#   user   system  elapsed   # 
# 14.49783  0.07785 14.70043 #
##############################

# CV
t1 <- proc.time()
set.seed(2018)
cv <- XGB_docv(param0, 1000) # returns just the best iteration; cv = 120
t1 <- proc.time() - t1


# create ensemble for submission
ensemble <- rep(0, nrow(test))

cv <- round(cv * 1.5)
cat("Calculated rounds:", cv, " Starting ensemble\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results
for (i in 1:10) {
    print(i)
    set.seed(i + 2017)
    p <- XGB_doTest(param0, cv) 
    # use 40% to 50% more than the best iter rounds from your cross-fold number.
    # as you have another 50% training data now, which gives longer optimal training time
    ensemble <- ensemble + p
}

Index_train <- !(is.na(Alldf$target))
write_submission(file = paste(ParamConfig$subm_dir, 'sub-B1-16-3-9.csv', sep = ''),
                 pred = ensemble/i)
# do a search for optimize the parameters

##########################################################################
# Parameter optimization with MLR
##########################################################################

train_task <- makeClassifTask(data = Alldf[!is.na(target), -c('ID'), with = FALSE],
                              target = 'target')
##########################
# CV & resampling scheme #
##########################
res_desc <- makeResampleDesc('RepCV', folds = 3, reps = 3)

xgb_lrn <- makeLearner('classif.xgboost', predict.type = 'prob')

xgb_lrn$par.vals <- list( # parameters not for tuning
    "objective"  = "binary:logistic",
    "eval_metric" = XGB_logloss,
    #"eta" = 0.05,
    #"subsample" = 0.9,
    #"colsample_bytree" = 0.8,
    #"min_child_weight" = 1,
    #"max_depth" = 10,
    nthread = 8,
    maximize = FALSE#,
    #nrounds = 4
)

########################################
# Search Method for Optimal Parameters #
########################################

ps_set <- makeParamSet(
    makeNumericParam('eta', lower = 0.02, upper = 0.2),
    makeNumericParam('gamma', lower = 0, upper = 1),
    makeNumericParam('subsample', lower = 0.5, upper = 1),
    makeNumericParam('colsample_bytree', lower = 0.5, upper = 1),
    makeDiscreteParam('max_depth', values = c(8, 10, 12, 15)),
    makeDiscreteParam('min_child_weight', values = c(1, 5, 10)),
    makeIntegerParam('nrounds', lower = 120, upper = 300)
)

opt_par_ctrl <- makeTuneControlRandom(budget = 20, maxit = 20)

# tune 
# in the future add a sink to a file to keep track of
# the progress

#################################                               
#     user    system   elapsed  #
# 1999.4273    7.3444 2237.9516 #
################################# 
# about 2 days
t1 <- proc.time()
set.seed(78)
xgb_tuned <- tuneParams(xgb_lrn, task = train_task, resampling = res_desc,
                        par.set = ps_set, control = opt_par_ctrl,
                       measures = list(MLR_logloss, auc, acc, fdr))
t1 <- proc.time() - t1

# save
save(xgb_tuned, file = paste(ParamConfig$output_dir, 'xgb-tuned-16-3-12.RData', sep = ''))

#############################################################################################
# Tune result:
# Op. pars: eta=0.0313; gamma=0.981; subsample=0.711; colsample_bytree=0.609;
# max_depth=10; min_child_weight=1; nrounds=157
# BinaryLogLoss.test.mean=0.467,auc.test.mean=0.755,acc.test.mean=0.782,fdr.test.mean=0.346  
#############################################################################################

# Retrain on new parameters; expect optimal rounds to 157
param1 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic"
        , "eval_metric" = "logloss"
        , "eta" = 0.03
        , "gamma" = 1
        , "subsample" = 0.7
        , "colsample_bytree" = 0.6
        , "min_child_weight" = 1
        , "max_depth" = 10
        ) 

train <- data.matrix(Alldf[!is.na(target), -c(1,2), with = FALSE])
train.label <- Alldf[!is.na(target), target]
test <- data.matrix(Alldf[is.na(target), -c(1,2), with = FALSE])

xgtrain = xgb.DMatrix(train, label = train.label, missing=NA)
xgtest = xgb.DMatrix(test, missing=NA)

########################################
#       user      system     elapsed   #
#  16.22185000  0.09016667 16.47276667 #
########################################
# CV
t1 <- proc.time()
set.seed(2018)
cv <- XGB_docv(param1, 300) # returns just the best iteration; cv = 182
t1 <- proc.time() - t1

# create ensembles for submission
ensemble0 <- rep(0, nrow(test))
ensemble1 <- rep(0, nrow(test))
cv1 <- round(cv * 1.5)

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results, use cv
for (i in 1:10) {
    print(i)
    set.seed(i + 2017)
    p <- XGB_doTest(param1, cv) 
    # use 40% to 50% more than the best iter rounds from your cross-fold number.
    # as you have another 50% training data now, which gives longer optimal training time
    ensemble0 <- ensemble0 + p
}

for (i in 1:10) { # use larger cv
    print(i)
    set.seed(i + 2017)
    p <- XGB_doTest(param1, cv1) 
    # use 40% to 50% more than the best iter rounds from your cross-fold number.
    # as you have another 50% training data now, which gives longer optimal training time
    ensemble1 <- ensemble1 + p
}

Index_train <- !(is.na(Alldf$target))
write_submission(file = paste(ParamConfig$subm_dir, 'sub-B2-16-3-12.csv', sep = ''),
                 pred = ensemble0/i)
write_submission(file = paste(ParamConfig$subm_dir, 'sub-B3-16-3-12.csv', sep = ''),
                 pred = ensemble1/i)

##########################################################################
# Redo on Data Imputed by MICE
##########################################################################

# Use features with MICE
Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv", sep = ''))

# Boruta data
load(paste(ParamConfig$output_dir, 'boruta-16-3-8.RData', sep = ''))
boruta.smry <- arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)), desc(medianImp))

# Remove features suggested by Boruta
boruta.to.remove <- c('v72', 'v62', 'v112', 'v107',
                      'v125', 'v75', 'v71', 'v91',
                      'v74', 'v52', 'v22', 'v3')

# remove other less significant variables
further.to.remove <- grep('(Nr|Imp|T\\w+|Num|RFreq|FFreq|It|Len)', boruta.smry$attr[206:258], value = TRUE)

highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128")

boruta.to.remove <- unique(c(boruta.to.remove, further.to.remove, highCorrRemovals))
Alldf[, c(boruta.to.remove) := NULL]

# transform to factors
num.to.fac <- intersect(c('v38', 'v62', 'v72', 'v129'), names(Alldf))
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)

# Retrain on new parameters; expect optimal rounds to 157
param1 <- list(
        # some generic, non specific params
        "objective"  = "binary:logistic"
        , "eval_metric" = "logloss"
        , "eta" = 0.03
        , "gamma" = 1
        , "subsample" = 0.7
        , "colsample_bytree" = 0.6
        , "min_child_weight" = 1
        , "max_depth" = 10
        ) 

train <- data.matrix(Alldf[!is.na(target), -c(1,2), with = FALSE])
train.label <- Alldf[!is.na(target), target]
test <- data.matrix(Alldf[is.na(target), -c(1,2), with = FALSE])

xgtrain = xgb.DMatrix(train, label = train.label, missing=NA)
xgtest = xgb.DMatrix(test, missing=NA)

########################################
#      user     system    elapsed      #
#  18.5873500  0.1128667 18.8214500    #
########################################
# CV
t1 <- proc.time()
set.seed(2018)
cv <- XGB_docv(param1, 300) # returns just the best iteration; cv = 213
t1 <- proc.time() - t1

# create ensembles for submission
ensemble0 <- rep(0, nrow(test))

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results, use cv
for (i in 1:10) {
    print(i)
    set.seed(i + 2017)
    p <- XGB_doTest(param1, cv) 
    # use 40% to 50% more than the best iter rounds from your cross-fold number.
    # as you have another 50% training data now, which gives longer optimal training time
    ensemble0 <- ensemble0 + p
}

Index_train <- !(is.na(Alldf$target))
write_submission(file = paste(ParamConfig$subm_dir, 'sub-B4-16-3-12.csv', sep = ''),
                 pred = ensemble0/i)

########################################################
# Try to improve the score reducing the # of variables #
########################################################

XGB_simple_wrapper <- function(data, index.train, seed1 = 433, seed2= 765, numFolds = 2, numRepeats = 2)
{
    set.seed(seed1)

    train <- data[index.train,]
    test <- data[!index.train,]
    # remove ID, target
    test.DM  <- xgb.DMatrix(data = data.matrix(test[, -c(1,2), with = FALSE]), missing = NA)

    numRounds <- numFolds * numRepeats
    CVvec <- rep(0, numRounds)
    out_char <- list()
    importances <- list()
    
    folds <- createMultiFolds(train[, target], k = numFolds, times = numRepeats)
    names(CVvec) <- names(folds)
    
    train.pred <- rep(0, nrow(train))
    test.pred <- rep(0, nrow(test))

    set.seed(seed2)
    for (ii in seq(1, length(folds)) ) {
        other.ids <- setdiff(1:nrow(train), folds[[ii]])

        cat('\n', 'Iteration', ii, 'of', numRounds, paste('(', round(ii/numRounds*100),
                                                    '%);', sep = ''),
            'Fold:', names(folds)[ii], '\n', sep = ' ')
        
        train.DM  <- xgb.DMatrix(data = data.matrix(train[-other.ids,-c(1,2), with = FALSE]), 
                                 label = train[-other.ids, target], missing = NA)
        other.DM  <- xgb.DMatrix(data = data.matrix(train[other.ids, -c(1,2), with = FALSE]), 
                               label = train[other.ids, target], missing = NA)

        # watchlist for CV
        wlist <- list(val = other.DM, train = train.DM)

        # XGB params
        param <- list(max.depth = 10, eta = 0.03, booster = "gbtree", gamma = 1,
                       subsample = 0.7, colsample_bytree = 0.6,
                      objective = "binary:logistic", eval_metric = XGB_logloss)
        
        # save messages to a character vector
        temp <- tempfile()
        sink(temp, split = TRUE) # split to keep also sending to stdout
        
        model1 <- xgb.train(params = param, data = train.DM, nrounds = 2000,
                            early.stop.round = 30,
                            nthread = 4, verbose = 1, print.every.n = 20,
                            missing = NA, watchlist = wlist, maximize = FALSE)
        sink()
        write(NULL, temp, append = TRUE) # to get a proper EOF
        out_char[[names(folds)[ii]]] <- readLines(temp)
        file.remove(temp)
        
        bestIter <- model1$bestInd # get best iteration
        CVvec[[names(folds)[ii]]] <- model1$bestScore
        importances[[names(folds)[ii]]] <- xgb.importance(feature_names = names(train)[-c(1,2)],
                                                          model = model1)
        local.pred <- predict(model1, newdata = other.DM, ntreelimit = bestIter)
        
        train.pred[other.ids] <- train.pred[other.ids] + local.pred
        test.pred <- test.pred + predict(model1, newdata = test.DM, ntreelimit = bestIter)
    }

    # average on rounds and repeats
    test.pred <- test.pred/numRounds
    train.pred <- train.pred/numRepeats
   
    retval = list(Train_pred = train.pred, Test_pred = test.pred, CV_vec = CVvec, CV_mean = mean(CVvec),
                  CV_sd = sd(CVvec), output_strings = out_char, VarImp = importances)
    return (retval)
}

# first run to find importances
Index_train <- !(is.na(Alldf$target))
out <- XGB_simple_wrapper(data = Alldf, index.train = Index_train, numFolds = 3, numRepeats = 3)

# find and save importances
VarImpList <- out$VarImp
totalImp <- data.table(Feature = VarImpList[[1]][, Feature])
for(i in 1:length(VarImpList)) {
    totalImp[, c(paste('Gain', i, sep = '')) := VarImpList[[i]][, Gain]]
}

totalImp[, Average := apply(.SD, 1, mean), .SDcols = grep('Gain', names(totalImp))]
VarImpList$Total <- totalImp[order(-Average), .(Feature, Average)]
save(VarImpList, file = paste(ParamConfig$output_dir, 'xgb-importances-16-3-13.RData', sep = ''))

# Retrain on 50 and 100 best features
best100 <- VarImpList$Total[1:100, Feature]
best50 <- VarImpList$Total[1:50, Feature]

#######################################
#     user     system    elapsed 
# 90.7258000  0.8051167 91.8334500
# out50 CV 0.4679041 +- 0.00123927
# out100 CV 0.4674438 +- 0.001385856  
#######################################

t1 <- proc.time()
out50 <- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', best50), with = FALSE],
                             index.train = Index_train, numFolds = 3, numRepeats = 3)
out100 <- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', best100), with = FALSE],
                             index.train = Index_train, numFolds = 3, numRepeats = 3)
t1 <- proc.time() - t1

#####################################
#     user     system    elapsed 
# 123.047983   1.318117 125.739233 
# outv56: 0.4665447 +- 0.002901691
# outv113: 0.4657721 +- 0.003217996
# outv79: 0.4658303 +- 0.003053101   
#####################################

# Try removing one of Imp features

t1 <- proc.time()
outv56 <- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', setdiff(best100, 'Imp_v56')), with = FALSE],
                             index.train = Index_train, numFolds = 5, numRepeats = 1)
outv113 <- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', setdiff(best100, 'Imp_v113')), with = FALSE],
                             index.train = Index_train, numFolds = 5, numRepeats = 1)

outv79 <- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', setdiff(best100, 'Imp_v79')), with = FALSE],
                             index.train = Index_train, numFolds = 5, numRepeats = 1)
t1 <- proc.time() - t1

# Remove all Imp features; adjust gamma on second round

##############################################
#      user     system    elapsed 
# 79.6165833  0.6713833 80.5979833
# outv_noImp 0.4674277 +- 0.003017155
# outv_noImp_newpar 0.4666602 +- 0.003049392  
##############################################

t1 <- proc.time()
outv_noImp <- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', setdiff(best100, grep('Imp_', best100,
                                                                                        value = TRUE))),
                                              with = FALSE],
                                 index.train = Index_train, numFolds = 5, numRepeats = 1)
# adjust gamma; subsample, etc...
XGB_simple_wrapper <- function(data, index.train, seed1 = 433, seed2= 765, numFolds = 2, numRepeats = 2)
{
    set.seed(seed1)

    train <- data[index.train,]
    test <- data[!index.train,]
    # remove ID, target
    test.DM  <- xgb.DMatrix(data = data.matrix(test[, -c(1,2), with = FALSE]), missing = NA)

    numRounds <- numFolds * numRepeats
    CVvec <- rep(0, numRounds)
    out_char <- list()
    importances <- list()
    
    folds <- createMultiFolds(train[, target], k = numFolds, times = numRepeats)
    names(CVvec) <- names(folds)
    
    train.pred <- rep(0, nrow(train))
    test.pred <- rep(0, nrow(test))

    set.seed(seed2)
    for (ii in seq(1, length(folds)) ) {
        other.ids <- setdiff(1:nrow(train), folds[[ii]])

        cat('\n', 'Iteration', ii, 'of', numRounds, paste('(', round(ii/numRounds*100),
                                                    '%);', sep = ''),
            'Fold:', names(folds)[ii], '\n', sep = ' ')
        
        train.DM  <- xgb.DMatrix(data = data.matrix(train[-other.ids,-c(1,2), with = FALSE]), 
                                 label = train[-other.ids, target], missing = NA)
        other.DM  <- xgb.DMatrix(data = data.matrix(train[other.ids, -c(1,2), with = FALSE]), 
                               label = train[other.ids, target], missing = NA)

        # watchlist for CV
        wlist <- list(val = other.DM, train = train.DM)

        # XGB params
        param <- list(max.depth = 10, eta = 0.03, booster = "gbtree", gamma = 0.5,
                       subsample = 0.8, colsample_bytree = 0.7,
                      objective = "binary:logistic", eval_metric = XGB_logloss)
        
        # save messages to a character vector
        temp <- tempfile()
        sink(temp, split = TRUE) # split to keep also sending to stdout
        
        model1 <- xgb.train(params = param, data = train.DM, nrounds = 2000,
                            early.stop.round = 30,
                            nthread = 4, verbose = 1, print.every.n = 20,
                            missing = NA, watchlist = wlist, maximize = FALSE)
        sink()
        write(NULL, temp, append = TRUE) # to get a proper EOF
        out_char[[names(folds)[ii]]] <- readLines(temp)
        file.remove(temp)
        
        bestIter <- model1$bestInd # get best iteration
        CVvec[[names(folds)[ii]]] <- model1$bestScore
        importances[[names(folds)[ii]]] <- xgb.importance(feature_names = names(train)[-c(1,2)],
                                                          model = model1)
        local.pred <- predict(model1, newdata = other.DM, ntreelimit = bestIter)
        
        train.pred[other.ids] <- train.pred[other.ids] + local.pred
        test.pred <- test.pred + predict(model1, newdata = test.DM, ntreelimit = bestIter)
    }

    # average on rounds and repeats
    test.pred <- test.pred/numRounds
    train.pred <- train.pred/numRepeats
   
    retval = list(Train_pred = train.pred, Test_pred = test.pred, CV_vec = CVvec, CV_mean = mean(CVvec),
                  CV_sd = sd(CVvec), output_strings = out_char, VarImp = importances)
    return (retval)
}

outv_noImp_newpar<- XGB_simple_wrapper(data = Alldf[, c('ID', 'target', setdiff(best100, grep('Imp_', best100,
                                                                                        value = TRUE))),
                                              with = FALSE],
                                 index.train = Index_train, numFolds = 5, numRepeats = 1)
t1 <- proc.time() - t1

##########################################################################
# Change eta -> 0.05, gamma = 0.5, reduce col to 0.6
# Try first all data
##########################################################################
# Use features with MICE
Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv", sep = ''))
# transform to factors
num.to.fac <- intersect(c('v38', 'v62', 'v72', 'v129'), names(Alldf))
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)

XGB_simple_wrapper <- function(data, index.train, seed1 = 433, seed2= 765, numFolds = 2, numRepeats = 2)
{
    set.seed(seed1)

    train <- data[index.train,]
    test <- data[!index.train,]
    # remove ID, target
    test.DM  <- xgb.DMatrix(data = data.matrix(test[, -c(1,2), with = FALSE]), missing = NA)

    numRounds <- numFolds * numRepeats
    CVvec <- rep(0, numRounds)
    out_char <- list()
    importances <- list()
    
    folds <- createMultiFolds(train[, target], k = numFolds, times = numRepeats)
    names(CVvec) <- names(folds)
    
    train.pred <- rep(0, nrow(train))
    test.pred <- rep(0, nrow(test))

    set.seed(seed2)
    for (ii in seq(1, length(folds)) ) {
        other.ids <- setdiff(1:nrow(train), folds[[ii]])

        cat('\n', 'Iteration', ii, 'of', numRounds, paste('(', round(ii/numRounds*100),
                                                    '%);', sep = ''),
            'Fold:', names(folds)[ii], '\n', sep = ' ')
        
        train.DM  <- xgb.DMatrix(data = data.matrix(train[-other.ids,-c(1,2), with = FALSE]), 
                                 label = train[-other.ids, target], missing = NA)
        other.DM  <- xgb.DMatrix(data = data.matrix(train[other.ids, -c(1,2), with = FALSE]), 
                               label = train[other.ids, target], missing = NA)

        # watchlist for CV
        wlist <- list(val = other.DM, train = train.DM)

        # XGB params
        param <- list(max.depth = 10, eta = 0.05, booster = "gbtree", gamma = 0.5,
                       subsample = 0.7, colsample_bytree = 0.6,
                      objective = "binary:logistic", eval_metric = XGB_logloss)
        
        # save messages to a character vector
        temp <- tempfile()
        sink(temp, split = TRUE) # split to keep also sending to stdout
        
        model1 <- xgb.train(params = param, data = train.DM, nrounds = 2000,
                            early.stop.round = 30,
                            nthread = 4, verbose = 1, print.every.n = 20,
                            missing = NA, watchlist = wlist, maximize = FALSE)
        sink()
        write(NULL, temp, append = TRUE) # to get a proper EOF
        out_char[[names(folds)[ii]]] <- readLines(temp)
        file.remove(temp)
        
        bestIter <- model1$bestInd # get best iteration
        CVvec[[names(folds)[ii]]] <- model1$bestScore
        importances[[names(folds)[ii]]] <- xgb.importance(feature_names = names(train)[-c(1,2)],
                                                          model = model1)
        local.pred <- predict(model1, newdata = other.DM, ntreelimit = bestIter)
        
        train.pred[other.ids] <- train.pred[other.ids] + local.pred
        test.pred <- test.pred + predict(model1, newdata = test.DM, ntreelimit = bestIter)
    }

    # average on rounds and repeats
    test.pred <- test.pred/numRounds
    train.pred <- train.pred/numRepeats
   
    retval = list(Train_pred = train.pred, Test_pred = test.pred, CV_vec = CVvec, CV_mean = mean(CVvec),
                  CV_sd = sd(CVvec), output_strings = out_char, VarImp = importances)
    return (retval)
}

# out_All_mT : Remove transformed data
# out_All_mboruta: Remove Boruta
# out_All_mboruta_mT: Remove transformed and Boruta

boruta.to.rem <- c('v72', 'v62', 'v112', 'v107',
                      'v125', 'v75', 'v71', 'v91',
                   'v74', 'v52', 'v22', 'v3')
Tvars <- grep('^T', names(Alldf), value = TRUE)

################################################
#      user     system    elapsed 
# 173.143350   1.475617 175.891683
# out_All: 0.4647981 +- 0.003568711
# out_All_mT: 0.4641886 +- 0.003514388
# out_All_mboruta: 0.4659295 +- 0.002944755
# out_All_mboruta_mT: 0.4649105 +- 0.003125593
################################################


t1 <- proc.time()
out_All <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v113'), with = FALSE],
                              index.train = Index_train, numFolds = 5, numRepeats = 1)
out_All_mT <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v113', Tvars), with = FALSE],
                                 index.train = Index_train, numFolds = 5, numRepeats = 1)
out_All_mboruta <- XGB_simple_wrapper(data = Alldf[, setdiff(names(Alldf), c('Imp_v113',
                                                                             boruta.to.rem)), with = FALSE],
                                      index.train = Index_train, numFolds = 5, numRepeats = 1)
out_All_mboruta_mT<- XGB_simple_wrapper(data = Alldf[, setdiff(names(Alldf), c('Imp_v113',
                                                                               boruta.to.rem,
                                                                               Tvars)), with = FALSE],
                                      index.train = Index_train, numFolds = 5, numRepeats = 1)
t1 <- proc.time() - t1

# Finally train on All and All_mt

#######################################
#     user    system   elapsed 
# 1245.9435   13.0174 1277.7119 
# out_All: 0.4637705 +- 0.004073895
# out_All_mT: 0.4635296 +- 0.00394731
#######################################

t1 <- proc.time()
out_All <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v113'), with = FALSE],
                              index.train = Index_train, numFolds = 10, numRepeats = 1)
out_All_mT <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v113', Tvars), with = FALSE],
                                 index.train = Index_train, numFolds = 10, numRepeats = 1)
t1 <- proc.time() - t1

Index_train <- !(is.na(Alldf$target))
write_submission(file = paste(ParamConfig$subm_dir, 'sub-B5-16-3-13.csv', sep = ''),
                 pred = out_All$Test_pred)
write_submission(file = paste(ParamConfig$subm_dir, 'sub-B6-16-3-13.csv', sep = ''),
                 pred = out_All_mT$Test_pred)

# find and save importances
VarImpListAll <- out_All$VarImp
totalImp <- data.table(Feature = VarImpListAll[[1]][, Feature])
for(i in 1:length(VarImpListAll)) {
    totalImp[, c(paste('Gain', i, sep = '')) := VarImpListAll[[i]][, Gain]]
}

totalImp[, Average := apply(.SD, 1, mean), .SDcols = grep('Gain', names(totalImp))]
VarImpListAll$Total <- totalImp[order(-Average), .(Feature, Average)]

VarImpListAllmT <- out_All_mT$VarImp
totalImp <- data.table(Feature = VarImpListAllmT[[1]][, Feature])
for(i in 1:length(VarImpList)) {
    totalImp[, c(paste('Gain', i, sep = '')) := VarImpListAllmT[[i]][, Gain]]
}

totalImp[, Average := apply(.SD, 1, mean), .SDcols = grep('Gain', names(totalImp))]
VarImpListAllmT$Total <- totalImp[order(-Average), .(Feature, Average)]


save(VarImpListAll, VarImpListAllmT,
     file = paste(ParamConfig$output_dir, 'xgb-importances-16-3-14.RData', sep = ''))
