##########################################################################
# Simple Imputation Methods
##########################################################################

library(bit64)
library(data.table) # more efficient
library(mlr)
library(ggplot2)
library(mice)
library(caret)


setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code/model')
source('../param_config.R') 
source('../utils.R')
source('../impute_utils.R')

#############
# load data #
############# 
Alldf_num <- fread(paste(ParamConfig$feat_dir, "all-numeric-raw_trans-16-2-21.csv", sep = ''))
Alldf_fac <- fread(paste(ParamConfig$feat_dir, "all-factor-genfea-16-2-29.csv", sep = ''),
                   colClasses = list(character = 32)) # to placate an error message
load(paste(ParamConfig$output_dir, 'raw-summaries-16-2-19.RData', sep = ''))

#####################################
# merge and impute NULL chars -> "" #
#####################################
Alldf <- merge(Alldf_num, Alldf_fac[, -c(2), with = FALSE], by = 'ID') # only one target
rm(Alldf_num, Alldf_fac)
system.time(impobj.Alldf_imputed <- impute(Alldf[, -c(1,2), with = FALSE], classes = list(
                                                                       character = imputeConstant(""))))#
Alldf_charimputed <- impobj.Alldf_imputed$data
Alldf_charimputed <- cbind(Alldf[, c(1,2), with = FALSE], Alldf_charimputed) # ! results in data.table

################################
# construct correlation matrix #
################################ 

# remove variables correlated with response
resp.to.remove <- c('ID', 'target')
imp.to.remove <- grep('Imp_+', names(Alldf), value = TRUE)

# char -> factor
# transform to factors
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(Alldf_charimputed, fcol = c(3:ncol(Alldf_charimputed)), extra = num.to.fac)
Alldf_charimputed.corr <- correlize(Alldf_charimputed[, -c(resp.to.remove, imp.to.remove), with = FALSE]) # exclude ID
save(Alldf_charimputed.corr, file = paste(ParamConfig$output_dir, 'transf-correlations-16-3-2.RData', sep = ''))

###################################################
# Impute by median the numeric variables and save #
################################################### 
median_impobj.Alldf_imputed <- impute(Alldf_charimputed[, -c(1,2), with = FALSE],
                                                  classes = list(numeric = imputeMedian()))
                                                                      
Alldf_medianimputed <- median_impobj.Alldf_imputed$data
Alldf_medianimputed <- cbind(Alldf_charimputed[, c(1,2), with = FALSE], Alldf_medianimputed) # ! results in data.table
write.table(Alldf_medianimputed, paste(ParamConfig$feat_dir, "imputed-B1-16-3-4.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE, quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-B1-16-3-4.csv",
                                             sep = ''))$size/1024^2),"\n")
rm(Alldf_medianimputed)

##########################################################################
# Do a Random Imputation
##########################################################################
NAlist <- names(which(sapply(Alldf_charimputed, function(x) sum(is.na(x))) > 0))
NAlist <- setdiff(NAlist, c('ID', 'target'))

set.seed(321)
Alldf_randomimputed <- random_impute(Alldf_charimputed, cols = NAlist)

write.table(Alldf_randomimputed, paste(ParamConfig$feat_dir, "imputed-B2-16-3-4.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE, quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-B2-16-3-4.csv",
                                             sep = ''))$size/1024^2),"\n")
rm(Alldf_randomimputed)

##########################################################################
# Impute by MICE
##########################################################################
# load correlations computed above
load(paste(ParamConfig$output_dir, 'transf-correlations-16-3-2.RData', sep = ''))
# select 25 most correlated predictors for each variable
# reduces computing time; some are further eliminated when we eliminate double cols
best_sel <- bestN_mice_pred(Alldf_charimputed.corr$`num-fac`, N = 25)

# Construct copy; remove doubled cols and Imp+ which are related to the response
NAdoubled <- grep('T+', NAlist, value = TRUE)
imp.to.remove <- grep('Imp_+', names(Alldf), value = TRUE)
Alldf_mice <- Alldf[, -c('ID', 'target', NAdoubled, imp.to.remove), with = FALSE]

# dry run to construct the predMatrix
impdry <- mice(Alldf_mice[1:1000, ], maxit = 0)
predMatrix <- impdry$predictorMatrix
rm(impdry)

# modify predMatrix with best_sel
rcolnms <- colnames(best_sel)
rcolnms <- setdiff(rcolnms, NAdoubled)
predMatrix[rcolnms, rcolnms] <- best_sel[rcolnms, rcolnms]


# crete folds for subdivision
set.seed(47)
permuted <- sample(1:dim(Alldf_mice)[1])
folds <- list()
nFolds <- 8
indx_vec <- round(seq(1, length(permuted), length.out = nFolds + 1))
for(i in 1:(length(indx_vec)-1)) {
    folds[[i]] <- permuted[c(indx_vec[i]:indx_vec[i+1])]
}

# Global NA list
glob_NAlist <- list()
for(i in names(Alldf_mice)) {
    glob_NAlist[[i]] <- which(is.na(Alldf_mice[[i]]))
}

# process individual folders

time_list <- list() # about 88 mins per round
mRepeats <- 3
mIters <- 3
for(ifold in 1:nFolds) { ### fix 3 -> 1
    cat('Mice imputing ', ifold, '-th fold (', round(ifold/nFolds*100), '%)\n')
    
    t1 <- proc.time()
    imp <- mice(Alldf_mice[folds[[ifold]],], m = mRepeats, pred = predMatrix, maxit = mIters,
                seed = 37, visitSequence = 'monotone') # m = 1 generate a single imputation
    t1 <- proc.time() - t1
    time_list[[ifold]] <- t1

    impnms <- names(which(unlist(lapply(imp$imp, is.data.frame))))
    for(icol in impnms) {
        Alldf_mice[intersect(folds[[ifold]], glob_NAlist[[icol]]), icol] <- apply(imp$imp[[icol]], 1, mean)
    }

    # print pictures of convergence
    j = 1
    pdf(paste(ParamConfig$fig_dir, paste('mice-convergence-', ifold, '.pdf', sep = ''), sep = ''),
        width = 8, height = 8)
    while(j <= length(impnms)) {
        end <- min(j+2, length(impnms))
        print(plot(imp, impnms[j:end]))
        if(end == length(impnms)) break
        j <- j + 3
    }
    dev.off()
                   
    # backup step
    #save(Alldf_mice, file = paste('Backup', ifold, '.RData', sep = ''))

    # free resources
    rm(imp)
    gc()
}
#cp1 <- copy(Alldf_mice) # for safety

#######################################
# Get names of variables to transform #
#######################################
ys.names <- grep('Ts_+', NAdoubled, value = TRUE)
ys.source <- na.omit(sub('Ts_+(\\w+)', '\\1', ys.names))
ys.source <- as.character(ys.source)

yl.names <- grep('Tl_+', NAdoubled, value = TRUE)
yl.source <- na.omit(sub('Tl_+(\\w+)', '\\1', yl.names))
yl.source <- as.character(yl.source)

yj.names <- grep('Tyj_+', NAdoubled, value = TRUE)
yj.source <- na.omit(sub('Tyj_+(\\w+)', '\\1', yj.names))
yj.source <- as.character(yj.source)

###############################
# Transform according to type #
############################### 
Alldf_mice[, c(ys.names) := ifelse(is.na(log(.SD)), -40, log(v23)), .SDcols = ys.source]
Alldf_mice[, c(yl.names) := lapply(.SD, function(x) log(1+x)), .SDcols = yl.source]
yj.scaler <- preProcess(Alldf_mice[, yj.source, with = FALSE], method = c('YeoJohnson'))
Alldf_mice[, c(yj.names) := predict(yj.scaler, Alldf_mice[, yj.source, with = FALSE])] # will work??

# rescale new variables
new.names <- c(ys.names, yl.names, yj.names)
range.scaler <- preProcess(Alldf_mice[, new.names, with = FALSE], method = c('range'))
Alldf_mice[, c(new.names) := predict(range.scaler, Alldf_mice[, new.names, with = FALSE])]

# bind and save
#cp1 <- copy(Alldf_mice) # for safety
Alldf_mice <- cbind(Alldf_charimputed[, .(ID, target)], Alldf_mice, Alldf_charimputed[, imp.to.remove,
                                                                                      with = FALSE])
system.time(impobj.Alldf_mice <- impute(Alldf_mice[, -c(1,2), with = FALSE], classes = list(
                                                                       character = imputeConstant(""))))#
Alldf_mice <- impobj.Alldf_mice$data
write.table(Alldf_mice, paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE, quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv",
                                             sep = ''))$size/1024^2),"\n")

#Fix a mistake about not writing ID and target                                      
Alldf <- fread(paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv", sep = ''))
Alldf_old <- fread(paste(ParamConfig$feat_dir, "imputed-B2-16-3-4.csv", sep = ''))

Alldf <- cbind(Alldf_old[, c('ID', 'target'), with = FALSE], Alldf)
write.table(Alldf, paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE, quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-B3-16-3-6.csv",
                                             sep = ''))$size/1024^2),"\n")

##########################################################################
## Code for selecting best models
##########################################################################
M <-  bestN_mice_pred(Alldf_charimputed.corr$`num-fac`, N = 25)

# Avoid that a transformed variable interacts with the untransformed one
for(i in rownames(M)) {
    nms <- paste(c('Ts_', 'Tyj_', 'Tl_'), i, sep = '')
    for(j in nms) {
        if(j %in% rownames(M)) M[i, j] <- 0
    }
}

dt <- random_impute(Alldf_charimputed[1:1000,], cols = NAlist)

bestM_RReg_Impute <- function(df, M, rowvec = NULL) { # helper function: select best models for vars to impute
    step_modl <- list()
    warn_list <- list()
    err_list <- list()
    if(is.null(rowvec)) rowvec <- rownames(M)
    
    for(i in 1:length(rowvec)) {
        tgt <- rowvec[i]
        expr <- paste(names(which(M[tgt, ] == 1)), collapse = ' + ')
        form <- as.formula(paste(tgt, ' ~ ', expr))

        cat('Best model for ', tgt, ' (', round(i/length(rowvec)*100), '%)\n')
        
        tryCatch( {
            rawmodl <- lm(form, data = df)
            reduced.model <- step(rawmodl, direction="backward", trace = 0)
            step_modl[[tgt]] <- as.formula(as.character(reduced.model$call)[2])
        }, warning = function(war) {
            warn_list <- c(warn_list, message(war))
        }, error = function(err) {
            err_list <- c(err_list, message(err))
        })
        
    }
    
    out <- list(step_modl, warn_list, err_list)

    out
}

out <- bestM_RReg_Impute(dt, M, rowvec = NAlist)
########
function(dt, to.impute, M, miter = 10, seed = 22) # ass: to.impute contains all columns with NAs
    dt <- copy(dt)
    if(is.numeric(to.impute)) to.impute <- names(dt)[to.impute]
has.NAs <- apply(is.na(dt), 2, any)
has.NAs <- names(which(has.NAs))

positions.NAs <- list()
for(i in 1:length(has.NAs)) {
    positions.NAs[[has.NAs[i]]] <- which(is.na(dt[[has.NAs[i]]]))
}

cat('Preliminary Random Imputation\n')
set.seed(seed)
dt <- random_impute(dt, has.NAs) ### function to supply

time_list <- list()
t1 <- proc.time()
modellist <- bestM_RReg_Impute(dt, M, rowvec = has.NAs)
t1 <- t1 - proc.time()
time_list[[1]] <- t1

modellist <- modellist[[1]] # select only models; should check for errors

error_M <- matrix(0, nrow = miter - 1, ncol = length(has.NAs))
colnames(error_M) <- has.NAs

for(k in 1:miter) {
    cat('Iteration ', k, ' (', round(k/miter*100), '%)\n')
    for(ij in has.NAs) {
        if(ij %in% names(modellist)) {
            cat(ij, ' ')
            old <- dt[positions.NAs[[ij]], ij]
    model <- lm(modellist[[ij]], data = dt)
    preds <- predict(model, dt[positions.NAs[[ij]],])
    dt[positions.NAs[[ij]], ij] <- rnorm(length(positions.NAs[[ij]]),
                                         preds, summary(model)$sigma)
        if(k > 1) error_M[k - 1, ij] <- mean(abs(dt[positions.NAs[[ij]], ij] - old))
        }
    }
    cat('\n')
}

# return imputed and list of errors



df <- copy(Alldf_imputed)
df[, c('ID', 'target') := NULL]
miss_list <- list()

for(i in 1:ncol(df)) {
    miss_list[[i]] <- is.na(df[[i]])
}

for(i in 1:ncol(df)) {df[[i]] <- Random_Imputation(df[[i]])}
load(paste(ParamConfig$output_dir, 'all-raw-correlations-16-2-18.RData', sep = ''))
pred.nms <- names(which(Alldf.corr$`num-fac`[2,]>0.05)[-1])
pred.nms <- names(df)[which(pred.nms %in% names(df))][-1]
ll <- paste('v1', sep = ' ~ ', paste(pred.nms, collapse = " + "))
mod1 <- lm(as.formula(ll), data = df)
reduced.model <- step(mod1, direction="backward", trace = 0)

