##########################################################################
# This file creates NAcounts, splits numerical and factors
# Transforms some numerical variables
##########################################################################

library(bit64)
library(data.table) # more efficient
library(caret)

setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code')
source('param_config.R') 
source('utils.R')

# read data
Testdf <- fread(paste(ParamConfig$data_dir, 'test.csv', sep =''))
Traindf <- fread(paste(ParamConfig$data_dir, 'train.csv', sep =''))
Testdf$target <- NA

# join train-test
Alldf <- rbind(Traindf, Testdf)
Alldf.coltype <- Alldf[, sapply(.SD, class)]

# Convert to factors and numeric
cols.to.num <- which(Alldf.coltype == 'integer')[-1] # Omit Id
cols.to.fac <- which(Alldf.coltype == 'character')
Alldf[, c(cols.to.num) := lapply(.SD, as.numeric), .SDcols = cols.to.num]
Alldf[, c(cols.to.fac) := lapply(.SD, factor), .SDcols = cols.to.fac]
# Some factors contains NAs; set '' to NA
Alldf.coltype <- Alldf[, sapply(.SD, class)]
col.fac <- which(Alldf.coltype == 'factor')
fac.na <- Alldf[, sapply(.SD, function(x) levels(x)[1]), .SDcols = col.fac]
fac.na <- which(fac.na == '')
for(i in 1:length(fac.na)) {
    levels(Alldf[[names(fac.na)[i]]])[1] <- NA
}

# free resources
rm(Testdf, Traindf)
gc()

################################################
# transform factors which were read as numeric #
################################################

tonumeric <- c('v38', 'v62', 'v72', 'v129') # see end of 0-exploratory.R
Alldf[, (tonumeric) := lapply(.SD, factor), .SDcols = tonumeric]

############################################
# Create NA freqs, Zeroes and Below Zeroes #
############################################ 
Alldf.ncol <- ncol(Alldf) - 2 # -2 to exclude ID, target
Alldf[, NAfreq := apply(.SD, 1, function(x) sum(is.na(x))/Alldf.ncol)]

Alldf.coltype <- Alldf[, sapply(.SD, class)]
col.num <- which(Alldf.coltype == 'numeric')
# remove 'target' and 'NAfreq' for following computations
col.num <- col.num[-c(which(names(col.num) %in% c('target', 'NAfreq')))]
col.fac <- which(Alldf.coltype == 'factor')
Alldf[, NAfacfreq := apply(.SD, 1, function(x) sum(is.na(x))/length(col.fac)),
      .SDcols = col.fac] # the zero frequency suggested on the forums might apply to empty factor levels
Alldf[, Below0freq := apply(.SD, 1, function(x) sum( na.omit(x) < 0)/Alldf.ncol),
      .SDcols = col.num] 
############################################
# separate and write numerical and factors #
############################################
Alldf.coltype <- Alldf[, sapply(.SD, class)]
col.num <- which(Alldf.coltype == 'numeric')
All_num_df <- Alldf[, c('ID', names(col.num)), with = FALSE]
All_fac_df <- Alldf[, c('ID', 'target', names(col.fac)), with = FALSE]

write.table(All_num_df, paste(ParamConfig$feat_dir, "all-numeric-raw-16-2-21.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE, quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-numeric-raw-16-2-21.csv",
                                             sep = ''))$size/1024^2),"\n")

write.table(All_fac_df, paste(ParamConfig$feat_dir, "all-factor-raw-16-2-21.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-factor-raw-16-2-21.csv",
                                             sep = ''))$size/1024^2),"\n")
# free resources
rm(Alldf)
gc()

#################################
# Transform numerical variables #
#################################

# special transformation to v23
# v23 -> log, but if NA generated to -40
v23.indx <- which(!is.na(All_num_df$v23))
All_num_df[v23.indx, `:=`(Ts_v23 = ifelse(is.na(log(v23)), -40, log(v23)))] # Ts = transformed + special
#All_num_df[v23.indx, ][Ts_v23 == NA, Ts_v23 := -40] # check that what is left is NA

# transform to log
load(file = paste(ParamConfig$output_dir, 'fea-num-trans-16-2-20.RData', sep = ''))
log.indx <- which(ftransvec == 'log(1+x)')
log.names <- paste('Tl_', names(log.indx), sep = '')
All_num_df[, c(log.names) := lapply(.SD, function(x) log(1+x)), .SDcols = names(log.indx)]

# YeoJohnson transform
yj.indx <- which(ftransvec == 'YeoJohnson')
yj.names <- paste('Tyj_', names(yj.indx), sep = '')
yj.scaler <- preProcess(All_num_df[, names(yj.indx), with = FALSE], method = c('YeoJohnson'))
All_num_df[, c(yj.names) := predict(yj.scaler, All_num_df[, names(yj.indx), with = FALSE])] # will work??

# range transform # do not do freqs!!!
freq.indx <- grep('([[:alnum:]]*)(freq+)', names(All_num_df), value = TRUE)
range.scaler <- preProcess(All_num_df[, -c('ID', 'target', freq.indx), with = FALSE], method = c('range'))
All_num_df[, c(setdiff(names(All_num_df), c('ID', 'target', freq.indx)))
           := predict(range.scaler, All_num_df[, c(setdiff(names(All_num_df), c('ID', 'target', freq.indx))),
                                               with = FALSE]), with = FALSE]

# Write 3 numeric data tables: with transformed, only rescaled and rescaled + transformed
# We do not rescale frequencies
trans.indx <- grep('T(\\w*)_', names(All_num_df), value = TRUE)
nontrans.indx <- c('ID', 'target', names(which(ftransvec == 'rescale')),
                   'v82', freq.indx)

write.table(All_num_df, paste(ParamConfig$feat_dir, "all-numeric-raw_trans-16-2-21.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-numeric-raw_trans-16-2-21.csv",
                                             sep = ''))$size/1024^2),"\n")

write.table(All_num_df[, c(nontrans.indx, trans.indx), with = FALSE],
            paste(ParamConfig$feat_dir, "all-numeric-trans-16-2-21.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-numeric-trans-16-2-21.csv",
                                             sep = ''))$size/1024^2),"\n")

write.table(All_num_df[, -c(trans.indx), with = FALSE],
            paste(ParamConfig$feat_dir, "all-numeric-raw-16-2-21.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE)
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-numeric-raw-16-2-21.csv",
                                             sep = ''))$size/1024^2),"\n")

