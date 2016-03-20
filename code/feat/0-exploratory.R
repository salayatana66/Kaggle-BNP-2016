##########################################################################
#                This file makes an exploratory data analysis
##########################################################################


library(bit64)
library(data.table) # more efficient
library(corrplot) # correlation plots
library(caret)
library(ggplot2)

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
Alldf[, (cols.to.num) := lapply(.SD, as.numeric), .SDcols = cols.to.num]
Alldf[, (cols.to.fac) := lapply(.SD, factor), .SDcols = cols.to.fac]

# construct correlation matrix
Alldf.corr <- correlize(Alldf[, -1, with = FALSE]) # exclude ID
save(Alldf.corr, file = paste(ParamConfig$output_dir, 'all-raw-correlations-16-2-18.RData', sep = ''))

# update train-test columns
train.indx <- which(!is.na(Alldf$target))
Traindf <- Alldf[train.indx, ]
Testdf <- Alldf[-train.indx, ]

##########################
# visualize correlations #
##########################
load(paste(ParamConfig$output_dir, 'all-raw-correlations-16-2-18.RData', sep = ''))
M <- Alldf.corr$`num-fac`
M[is.na(M)] <- 0

# loop since many columns
vrow <- seq(1, 132, by = 20)
vcol <- seq(1, 132, by = 20)

for(i in 1:(length(vrow)-1)) {
    for(j in 1:(length(vcol)-1)) {
        x11()
        corrplot(M[vrow[i]:vrow[i+1],
                   vcol[j]:vcol[j+1]], method = 'ellipse', order = 'hclust')
    }
}

####################
# create summaries #
####################
train_summary <- summarize_df(Traindf)
test_summary <- summarize_df(Testdf, test = TRUE)
all_summary <- summarize_df(Alldf)

train_numsummary <- summarize_numeric_cols(Traindf)
test_numsummary <- summarize_numeric_cols(Testdf)
all_numsummary <- summarize_numeric_cols(Alldf)

save(train_summary, test_summary, all_summary,
     train_numsummary, test_numsummary, all_numsummary,
     file = paste(ParamConfig$output_dir, 'raw-summaries-16-2-19.RData', sep = ''))

####################################
# select numerical transformations #
####################################
ftransvec <- select_num_transf(Alldf[, -c(1,2), with = FALSE])
save(ftransvec,
     file = paste(ParamConfig$output_dir, 'fea-num-trans-16-2-20.RData', sep = ''))

load(file = paste(ParamConfig$output_dir, 'fea-num-trans-16-2-20.RData', sep = ''))
other.indx <- which(ftransvec == 'other')

# v23: create a feature with log(...) and values which entailed NA set to -40
# v38: transform to categorical
# v62: transform to categorical
# v72: transform to categorical
# v82: leave as it is
# v129: to categorical
