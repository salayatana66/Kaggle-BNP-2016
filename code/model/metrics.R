##########################################################################
# Functions which define evaluation metrics
##########################################################################

#################################################################
# Binary Log-Loss rounded for Kaggle and its XGB implementation #
#################################################################

BinaryLogLoss <- function(actual_vec, prob_vec) {
    # regularize probs
    prob_vec <- sapply(prob_vec, function(x) max(min(x, 1-1e-15), 1e-15))
    
    -mean(actual_vec * log(prob_vec) + (1-actual_vec)*log(1-prob_vec))
}

XGB_logloss <- function(preds, dtrain) { # to be passed internally as eval_metric
    labels <- getinfo(dtrain, 'label')
    retval <- BinaryLogLoss(labels, preds)
    
    return(list(metric = 'Log-Loss', value = retval))
}

MLR_logloss_FUN <- function(task, model, pred, feats, m) {
    act_vec <- ifelse(pred$data$truth == '0', 0, 1)
    BinaryLogLoss(act_vec, pred$data$prob.1)
}

if('package:mlr' %in% search()) { # define only if MLR is loaded
    MLR_logloss <- makeMeasure(id = 'BinaryLogLoss', minimize = TRUE,
                               properties = c('classif', 'req.truth', 'req.prob'), # require true value and probability
                               fun = MLR_logloss_FUN, best = 0, worst = Inf)
}
