########################################
# Gelman - Data Analysis -- Multilevel #
########################################  
Random_Imputation <- function (a) {
    missing <- is.na(a)
    n.missing <- sum(missing)
    a.obs <- a[!missing]
    imputed <- a
    imputed[missing] <- sample (a.obs, n.missing, replace=TRUE)
    return (imputed)
}

##########################################################################
# Random imputation
##########################################################################

random_impute <- function(df, ...) UseMethod('random_impute')

random_impute.data.table <- function(df, cols) {
    if(is.numeric(cols)) cols <- names(df)[cols]

    N <- length(cols)
    imputed <- copy(df)
    for(i in 1:length(cols)) {
        icol <- cols[i]
        cat('Imputing ', icol, ' (', round(i/N*100), '%)\n', sep = '')
        imputed[[icol]] <- Random_Imputation(imputed[[icol]])
    }

    imputed
}

random_impute.data.frame <- function(df, cols) {
    if(is.numeric(cols)) cols <- names(df)[cols]

    N <- length(cols)
    
    for(i in 1:length(cols)) {
        icol <- cols[i]
        cat('Imputing ', icol, ' (', round(i/N*100), '%)\n', sep = '')
        df[[icol]] = Random_Imputation(df[[icol]])
    }

    df
}

##########################################################################
# Extract best N elements for prediction from 'Correlation' M
##########################################################################
bestN_mice_pred <- function(M, N = 25) {
    Ncol <- ncol(M)
    N <- min(N, Ncol)

    diag(M) <- 0 # remove self-self
    
    # helper to extract N-best cols
    bestN <- function(x) {
        out <- rep(0, Ncol)
        indx <- rev(order(abs(x)))[1:N]
        out[indx] <- 1

        out
    }

    out <- t(apply(M, 1, bestN)) # apply screws things up, need to transpose
    colnames(out) <- colnames(M)

    out 
}
