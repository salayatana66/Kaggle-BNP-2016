# Based on https://www.kaggle.com/jimthompson/bnp-paribas-cardif-claims-management/using-the-boruta-package-to-determine-fe/notebook

##########################################################################
# Boruta
##########################################################################

library(bit64)
library(data.table) # more efficient
library(caret)
library(Boruta)
library(dplyr)

setwd('/Users/schioand/leave_academia/kaggle/bnp-paribas/code/model')
source('../param_config.R') 
source('../utils.R')

# retrive sample data for analysis
train <- fread(paste(ParamConfig$feat_dir, "imputed-B2-16-3-4.csv", sep = ''))
train <- train[!is.na(train$target),]

# transform to factors
num.to.fac <- c('v38', 'v62', 'v72', 'v129')   
char_to_factors(train, fcol = c(3:ncol(train)), extra = num.to.fac)

###
# select random sample for analysis using caret createDataPartition() function
###
set.seed(123)
idx <- createDataPartition(train$target,p=0.01,list=FALSE)
sample.df <- train[idx,]

# eliminate ID, transform target to binary
df <- sample.df[,-c('ID', 'target'), with = FALSE]

set.seed(13)
t1 <- proc.time()
bor.results <- Boruta(df,factor(sample.df$target),
                   maxRuns=101,
                   doTrace=0)
t1 <- proc.time() - t1

#######################################
#       user      system     elapsed  # 
# 15.54256667  0.07851667  4.51755000 #
#######################################

# inspect results
save(bor.results, file = paste(ParamConfig$output_dir, 'boruta-16-3-8.RData', sep = ''))

print(bor.results)

sel_attrs <- getSelectedAttributes(bor.results)

# create plot
pdf(paste(ParamConfig$fig_dir, 'boruta-16-3-8.pdf', sep = ''),
        width = 8, height = 8)
plot(bor.results)
dev.off()

#Detailed results for each explanatory results

cat("\n\nAttribute importance details:\n")
options(width=125)
arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)), desc(medianImp))
