##########################################################################
# Simple XGB model 
##########################################################################

library(bit64)
library(data.table) # more efficient
library(xgboost)
library(caret)
library(ggplot2) # already loaded by caret

setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code/model')
source('../param_config.R') 
source('../utils.R')
source('metrics.R')

Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-A1-16-2-24.csv", sep = ''))
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)
Index_train <- !(is.na(Alldf$target))

# returns a list of: test-pred, train-pred, cvscore, output messages of xgb
# assumes data is a data.table and with col(1,2) = 'ID', 'target'
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
        param <- list(max.depth = 6, eta = 0.03, booster = "gbtree",
                       subsample = 0.6, colsample_bytree = 0.6,
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

out <- XGB_simple_wrapper(data = Alldf, index.train = Index_train, numFolds = 5, numRepeats = 3)

write_submission(file = paste(ParamConfig$subm_dir, 'sub-A1-16-2-27.csv', sep = ''),
                 pred = out$Test_pred)

# explore feature importance in the folds
for(ii in names(out$VarImp)) {
    x11()
    print(xgb.plot.importance(out$VarImp[[ii]][1:50,]))
}

##################################################################################
## Top variables
## Impv22 -> dangerous might overfit
## v50 & Tyj_v50; but v50 seems to make a better job
## The other Imps 56, 79, 113, which might make an overfit
## v10 better than Tyj_v10
## Nr_C, Nr_B might help
## Between the different fits there is a discrete agreement so we save one graph  
##################################################################################

p1 <- xgb.plot.importance(out$VarImp[['Fold3.Rep3']][1:50,])
ggsave(paste(ParamConfig$fig_dir, 'varimp-modelA1-16-2-27.pdf', sep = ''), plot = p1, width = 30,
       height = 30, units = 'cm')

# Imp_v22 likely to overfit
# Refit without Imp_v22
rm(out)
gc()
out <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v22'), with = FALSE],
                          index.train = Index_train, numFolds = 5, numRepeats = 3)

write_submission(file = paste(ParamConfig$subm_dir, 'sub-A2-16-2-27.csv', sep = ''),
                 pred = out$Test_pred)

# explore feature importance in the folds
for(ii in names(out$VarImp)) {
    x11()
    print(xgb.plot.importance(out$VarImp[[ii]][1:50,]))
}

##################################################################################
## Top variables
## v50 & Tyj_v50; but v50 seems to make a better job
## v66 is important
## The other Imps 56, 79, 113, which might make an overfit
## v10 better than Tyj_v10
## Nr_C, Nr_B might help
## Some new features are however helpful
## Between the different fits there is a discrete agreement so we save one graph  
##################################################################################

p2 <- xgb.plot.importance(out$VarImp[['Fold2.Rep3']][1:50,])
ggsave(paste(ParamConfig$fig_dir, 'varimp-modelA2-16-2-27.pdf', sep = ''), plot = p2, width = 30,
       height = 30, units = 'cm')

rm(out, Alldf)
gc()

# Fit on untrasformed data
Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-A3-16-2-25.csv", sep = ''))
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)
Index_train <- !(is.na(Alldf$target))

out <- XGB_simple_wrapper(data = Alldf, index.train = Index_train, numFolds = 5, numRepeats = 3)

write_submission(file = paste(ParamConfig$subm_dir, 'sub-A3-16-2-27.csv', sep = ''),
                 pred = out$Test_pred)

# explore feature importance in the folds
for(ii in names(out$VarImp)) {
    x11()
    print(xgb.plot.importance(out$VarImp[[ii]][1:50,]))
}

p3 <- xgb.plot.importance(out$VarImp[['Fold4.Rep2']][1:50,])
ggsave(paste(ParamConfig$fig_dir, 'varimp-modelA3-16-2-27.pdf', sep = ''), plot = p3, width = 30,
       height = 30, units = 'cm')

# VarImp_22 really produces overfitting
# CV on 5 folds and 3 repeats seems to go well on LB
# Try also 3 folds for the Public-Private split

####################################
# Fit on untrasformed data
# Remove Imp22
# Change cross-validation to 3 * 3  
####################################

Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-A3-16-2-25.csv", sep = ''))
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)
Index_train <- !(is.na(Alldf$target))

out <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v22'), with = FALSE],
                          index.train = Index_train, numFolds = 3, numRepeats = 3)

write_submission(file = paste(ParamConfig$subm_dir, 'sub-A4-16-2-27.csv', sep = ''),
                 pred = out$Test_pred)

p4 <- xgb.plot.importance(out$VarImp[['Fold1.Rep2']][1:50,])
ggsave(paste(ParamConfig$fig_dir, 'varimp-modelA4-16-2-27.pdf', sep = ''), plot = p4, width = 30,
       height = 30, units = 'cm')

rm(out)
gc()

# As above but remove all Imps...
imp.to.remove <- grep('Imp_+', names(Alldf), value = TRUE)
out <- XGB_simple_wrapper(data = Alldf[, -c(imp.to.remove), with = FALSE],
                          index.train = Index_train, numFolds = 3, numRepeats = 3)

write_submission(file = paste(ParamConfig$subm_dir, 'sub-A5-16-2-27.csv', sep = ''),
                 pred = out$Test_pred)

p5 <- xgb.plot.importance(out$VarImp[['Fold2.Rep2']][1:50,])
ggsave(paste(ParamConfig$fig_dir, 'varimp-modelA5-16-2-27.pdf', sep = ''), plot = p5, width = 30,
       height = 30, units = 'cm')

rm(out)
gc()
rm(Alldf)

######################################################
# Fit on raw data with entries with many NAs removed  
# Remove Imp_v22
# CV 5 * 3
######################################################
Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-A4-16-2-25.csv", sep = ''))
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(Alldf, fcol = c(3:ncol(Alldf)), extra = num.to.fac)
Index_train <- !(is.na(Alldf$target))

out <- XGB_simple_wrapper(data = Alldf[, -c('Imp_v22'), with = FALSE],
                          index.train = Index_train, numFolds = 3, numRepeats = 3)

write_submission(file = paste(ParamConfig$subm_dir, 'sub-A6-16-2-27.csv', sep = ''),
                 pred = out$Test_pred)

p6 <- xgb.plot.importance(out$VarImp[['Fold3.Rep2']][1:50,])
ggsave(paste(ParamConfig$fig_dir, 'varimp-modelA6-16-2-27.pdf', sep = ''), plot = p6, width = 30,
       height = 30, units = 'cm')
