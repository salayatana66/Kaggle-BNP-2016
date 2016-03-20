##########################################################################
# Transform and generate new factor variables
# It makes sense to impact some factors, but impacting v22 leads 
# to overfitting; thus replace other factors by frequencies in the data 
# (both relative and a factor version of them)
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

# For letter factors with more that 10 levels, EXCLUDING v22, create posteriors
# For numerical factors it is not needed
# Informally we call this technique 'impacting'
to.impact <- setdiff(names(which(all_summary$`fac-levels` > 10)), 'v22')
create_impacted(Alldf, tcol = 2, fcol = which(names(Alldf) %in% to.impact)) #!! some levels have not been
# shared between training and testing


# numerical columns to go to factors
num.to.fac <- c('v38', 'v62', 'v72', 'v129')
# keep a numeric copy 
Alldf[, paste('Num_', num.to.fac, sep = '') := .SD, .SDcols = num.to.fac]
Alldf.coltype <- Alldf[, sapply(.SD, class), .SDcols = c(3:ncol(Alldf))]
# convert to factors
tofac <- c(which(colnames(Alldf) %in% num.to.fac),
           which(colnames(Alldf) %in% names(which(Alldf.coltype == 'character'))))
Alldf[, c(tofac) := lapply(.SD, factor), .SDcols = tofac]

to.rfreqs <- names(which(all_summary$`fac-levels` >= 25))
factor_to_freqs(Alldf, cols = to.rfreqs, buckets = rep(25, length(to.rfreqs)))
Alldf[, c(to.rfreqs) := NULL]

write.table(Alldf, paste(ParamConfig$feat_dir, "all-factor-genfea-16-2-29.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE, quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "all-factor-genfea-16-2-29.csv",
                                             sep = ''))$size/1024^2),"\n")



