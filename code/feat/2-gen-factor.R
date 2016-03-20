##########################################################################
# Transform and generate new factor variables
##########################################################################

library(bit64)
library(data.table) # more efficient

setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code/feat')
source('../param_config.R') 
source('../utils.R')

Alldf <- fread(paste(ParamConfig$feat_dir, "all-factor-raw-16-2-21.csv", sep = ''))
load(paste(ParamConfig$output_dir, 'raw-summaries-16-2-19.RData', sep = ''))

# extrapolate letters from columns with strings of length > 1
extrapolate_char(Alldf, col = which(names(Alldf) %in% c('v22', 'v56', 'v113', 'v125')))
# add # of each letter A--Z across factors
letters_across(Alldf, fcol = c(3:ncol(Alldf)))

# For letter factors with more that 10 levels create posteriors
# For numerical factors it is not needed
to.impact <- names(which(all_summary$`fac-levels` > 10))
create_impacted(Alldf, tcol = 2, fcol = which(names(Alldf) %in% to.impact)) #!! some levels have not been
# shared between training and testing

####################################
# Summary of problematic columns    
# v22: train(18211), total(23420)
# v56: train(123), total(131)
# v71: train(9), total(12)
# v113: train(37), total(38) 
####################################

Alldf[, c(to.impact) := NULL]

# numerical columns to go to factors
num.to.fac <- c('v38', 'v62', 'v72', 'v129')
Alldf.coltype <- Alldf[, sapply(.SD, class), .SDcols = c(3:ncol(Alldf))]

tofac <- c(which(colnames(Alldf) %in% num.to.fac),
           which(colnames(Alldf) %in% names(which(Alldf.coltype == 'character'))))
Alldf[, c(tofac) := lapply(.SD, factor), .SDcols = tofac]

write.table(Alldf, paste(ParamConfig$feat_dir, "all-factor-genfea-16-2-24.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-factor-genfea-16-2-24.csv",
                                             sep = ''))$size/1024^2),"\n")

