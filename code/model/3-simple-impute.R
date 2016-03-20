##########################################################################
# Simple Imputation Methods
##########################################################################

library(bit64)
library(data.table) # more efficient
library(mlr)
library(ggplot2)

setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code/model')
source('../param_config.R') 
source('../utils.R')

##########################################################################
# Impute numeric with median, character with ""
##########################################################################

Alldf_num <- fread(paste(ParamConfig$feat_dir, "all-numeric-raw_trans-16-2-21.csv", sep = ''))
Alldf_fac <- fread(paste(ParamConfig$feat_dir, "all-factor-genfea-16-2-24.csv", sep = ''))

# merge
Alldf <- merge(Alldf_num, Alldf_fac[, -c(2), with = FALSE], by = 'ID') # only one target

# impute; works with data.table but $data is a data.frame
# time = 127
system.time(impobj.Alldf_imputed <- impute(Alldf[, -c(1,2), with = FALSE], classes = list(numeric = imputeMode(),
                                                                       character = imputeConstant("")))) #
Alldf_imputed <- impobj.Alldf_imputed$data
Alldf_imputed <- cbind(Alldf[, c(1,2), with = FALSE], Alldf_imputed) # ! results in data.table

# transform to factors
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(Alldf_imputed, fcol = c(3:ncol(Alldf_imputed)), extra = num.to.fac)

write.table(Alldf_imputed, paste(ParamConfig$feat_dir, "imputed-A1-16-2-24.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-A1-16-2-24.csv",
                                             sep = ''))$size/1024^2),"\n")

# create plots of distribution of NAfreq
p1 <- ggplot(Alldf_imputed, aes(x=NAfreq, y = ..density..))
p1 <- p1 + geom_histogram(binwidth=0.01, fill='cornsilk', colour="grey60", size=.2) + geom_line(stat='density')
ggsave(paste(ParamConfig$fig_dir, 'hist-NAfreq-16-2-25.pdf', sep = ''), plot = p1, width = 40, height = 20, units = 'cm')
# two spikes: imputing might be a good idea; maybe they put the NAs on purpose

binsize <- diff(range(Alldf_imputed$NAfreq))/15
p2 <- ggplot(Alldf_imputed, aes(x = NAfreq)) + geom_freqpoly(binwidth=binsize)
ggsave(paste(ParamConfig$fig_dir, 'polyfreq-NAfreq-16-2-25.pdf', sep = ''), plot = p2, width = 40, height = 20, units = 'cm')

# cut for NAs @0.778
indx <- Alldf_imputed$NAfreq <= 0.778
# write
write.table(Alldf_imputed[indx, ], paste(ParamConfig$feat_dir, "imputed-A2-16-2-25.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-A2-16-2-25.csv",
                                             sep = ''))$size/1024^2),"\n")

# Do with only original features
Alldf_num_orig <- fread(paste(ParamConfig$feat_dir, "all-numeric-raw-16-2-21.csv", sep = ''))
old_names <- intersect(names(Alldf_imputed), union(names(Alldf_num_orig), names(Alldf_fac)))
write.table(Alldf_imputed[, old_names, with = FALSE], paste(ParamConfig$feat_dir, "imputed-A3-16-2-25.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-A3-16-2-25.csv",
                                             sep = ''))$size/1024^2),"\n")


write.table(Alldf_imputed[indx, old_names, with = FALSE], paste(ParamConfig$feat_dir, "imputed-A4-16-2-25.csv",
                              sep = ''),
                              sep = ",", row.names=FALSE,quote=FALSE) 
cat("File size (MB):", round(file.info(paste(ParamConfig$feat_dir, "imputed-A4-16-2-25.csv",
                                             sep = ''))$size/1024^2),"\n")
