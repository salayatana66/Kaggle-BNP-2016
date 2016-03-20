##########################################################################
# Utilities for Exploratory Analysis and Feature generation
##########################################################################


# Compute Cramer's V; if chisq is unreliable MonteCarlo = #iterations in exact
Cramer_V <- function(x,y, MonteCarlo = 1000) {
    if(length(x)==1 || length(x) != length(y)) return(NA)
    
    chistat <- tryCatch(chisq.test(x, y, correct=FALSE), warning = function(w) w)

    # standard chisq unreliable
    if(is(chistat, 'warning')) {
        cat('Use Monte Carlo in chisq\n')
        chistat <- chisq.test(x, y, correct=FALSE, simulate.p.value = TRUE, B = MonteCarlo)
    }
    
    CV <- sqrt(chistat$statistic / (length(x) * (min(length(unique(x)),length(unique(y))) - 1)))
    return(as.numeric(CV))
}


##########################################################################
# Correlations between colums 
##########################################################################

# Return correlation list of columns; fac-fac: chisq, num-fac: r2-regression
correlize <- function(df, ...) UseMethod('correlize')

# levels_thresh: for categorical variables with many levels
correlize.data.frame <- function(df, levels_thresh = 100) {

    out <- list()

                                        # partition columns by type
    col.type <- sapply(df, class)
    col.num <- which(col.type %in% c('numeric', 'integer'))
    col.fac <- which(col.type == 'factor')
    len.fac <- length(col.fac)
    len.num <- length(col.num)

                                        # numeric correlation
    if(len.num >= 1) {
        cat('Compute numerical correlations\n')
        t1 <- system.time(out[['num-num']] <- cor(df[,col.num], use = 'pairwise.complete.obs'))
        cat('Time elapsed\n')
        print(t1)
    }

                                        # factor 'correlation' = Cramer_V
    if(len.fac >= 1) {
        cat('Compute factor correlations\n')
        t1 <- proc.time()
        mat.fac <- matrix(0, ncol = len.fac, nrow = len.fac)
        colnames(mat.fac) <- names(df)[col.fac]
        rownames(mat.fac) <- names(df)[col.fac]
        diag(mat.fac) <- 1 # perfect correlation
        
        for(i in 1:len.fac) { # loop
            cat('i = ', i, '\n', sep = '')
            j <- i + 1
            num_levels1 <- length(levels(df[[col.fac[i]]]))
            if(num_levels1 <= levels_thresh) {
                while(j <= len.fac) {
                    cat('j = ', j, '\n', sep = '')
                    obs <- complete.cases(df[[col.fac[i]]], df[[col.fac[j]]])
                    
                    num_levels2 <- length(levels(df[[col.fac[j]]]))
                    if(any(obs) && num_levels2 <= levels_thresh) {
                        cv <- Cramer_V(df[obs, col.fac[i]], df[obs, col.fac[j]])
                        mat.fac[i,j] <- cv
                        mat.fac[j, i] <- cv
                    } else {
                        cat('Assigning NA\n')
                        mat.fac[i,j] <- NA
                        mat.fac[j, i] <- NA
                        
                    }
                    j <- j + 1 # ! for loop to terminate :)
                }
            }
        }
        out[['fac-fac']] <- mat.fac
        t1 <- proc.time() - t1
        cat('Time elapsed\n')
        print(t1)
    }


                                        # num vs factor via r2 in linear regression
    if(len.fac >= 1 && len.num >= 1) {
        cat('Compute factor/numeric correlations\n')
        t1 <- proc.time()
        
        mat.tot <- matrix(0, ncol = len.num + len.fac, nrow = len.num + len.fac)
        colnames(mat.tot) <- c(names(df)[col.num], names(df)[col.fac])
        rownames(mat.tot) <- colnames(mat.tot)
        
        for(i in 1:len.num) { # loop
            cat('i = ', i, '\n', sep = '')
            for(j in 1:len.fac) {
                cat('j = ', j, '\n', sep = '')
                obs <- complete.cases(df[[col.num[i]]], df[[col.fac[j]]])
                num_levels <- length(levels(df[[col.fac[j]]]))
                if(any(obs) && num_levels <= levels_thresh) {
                    dd <- data.frame(y = df[obs, col.num[i]], x = df[obs, col.fac[j]])
                    cr <- summary(lm(y ~ x, data = dd))$adj.r.squared
                    mat.tot[i, j+len.num] <- cr
                    mat.tot[j+len.num, i] <- cr
                } else {
                    cat('Assigning NA\n')
                    mat.tot[i, j+len.num] <- NA
                    mat.tot[j+len.num, i] <- NA
                }
            }
        }
        mat.tot[c(1:len.num), c(1:len.num)] <- out[['num-num']]
        mat.tot[c((len.num+1):(len.num+len.fac)), c((len.num+1):(len.num+len.fac))] <- out[['fac-fac']]
        out[['num-fac']] <- mat.tot
        t1 <- proc.time() - t1
        cat('Time elapsed\n')
        print(t1)
    }

    out
}

correlize.data.table <- function(df, ...) {
    df <- copy(df) # deep copy
    class(df) <- 'data.frame'
    correlize.data.frame(df, ...)
}

##########################################################################
# summarize data  1st col = id, 2nd = target, test = T omits response
# target = 'factor', test = FALSE
##########################################################################

summarize_df <- function(df, ...) UseMethod('summarize_df')
summarize_df.default <- function(df, target = 'factor', test = FALSE) 
{
    out <- list()

    df.ncol <- ncol(df)
    df.nrow <- nrow(df)
    df.start <- 2 # start column
    if(!test) {
        df.start <- 3
        df.tgt <- df[, 2]
        if(target == 'factor') {
            df.tgt <- factor(df.tgt)
            out[['target-tot']] <- summary(df.tgt)
            out[['target-rel']] <- summary(df.tgt)/df.nrow
        } else {
            out[['target-smry']] <- summary(df.tgt)
        }
    }
    
    if(df.ncol > 2) {
        out[['col-class']] <- sapply(df[, c(df.start:df.ncol)], class) # preserves names
        out[['col-na-tot']] <- sapply(df[, c(df.start:df.ncol)], function(x) sum(is.na(x)))
        out[['col-na-rel']] <- out[['col-na-tot']]/df.nrow
    }

    col.fac <- which(out[['col-class']] == 'factor')
    col.fac <- col.fac + (df.start - 1)
    out[['fac-levels']] <- sapply(df[, col.fac], function(x) length(levels(x)))
    
    out
}

summarize_df.data.table <- function(df, target = 'factor', test = FALSE) # implementation for data.table
{
    df <- copy(df) # deep copy: df passed by reference
    out <- list()

    sum_NA <- function(x) sum(is.na(x))
    
    df.ncol <- ncol(df)
    df.nrow <- nrow(df)
    tgt.name <- colnames(df)[2]
    df.start <- 2 # start column
    if(!test) {
        df.start <- 3
        
        if(target == 'factor') {          
            df[, c(tgt.name) := lapply(.SD, factor), .SDcols = c(tgt.name)]
            out[['target-tot']] <- df[, sapply(.SD, summary), .SDcols = c(tgt.name)]
            out[['target-rel']] <- out[['target-tot']]/df.nrow
        } else {
            out[['target-smry']] <- df[, sapply(.SD, summary), .SDcols = c(tgt.name)]
        }
    }
    
    if(df.ncol > 2) {
        out[['col-class']] <- df[, sapply(.SD, class), .SDcols = c(df.start:df.ncol)]
        out[['col-na-tot']] <- df[, sapply(.SD, sum_NA), .SDcols = c(df.start:df.ncol)]
        out[['col-na-rel']] <- out[['col-na-tot']]/df.nrow
    }

    col.fac <- which(out[['col-class']] == 'factor')
    col.fac <- col.fac + (df.start - 1)
    out[['fac-levels']] <- df[, sapply(.SD, function(x) length(levels(x))),
                              .SDcols = col.fac]
    
    out
}

##########################################################################    
# Summarize numerical columns
##########################################################################
summarize_numeric_cols <- function(df, ...) UseMethod('summarize_numeric_cols')

summarize_numeric_cols.data.frame <- function(df) {

    create_summary <- function(x) {
        x.sum <- summary(x)
        class(x.sum) <- NULL # becomes a named vector
        x.sum <- c(x.sum, 'skewness' = moments::skewness(x, na.rm = TRUE),
                   'kurtosis' = moments::kurtosis(x, na.rm = TRUE),
                   'jarque-brera' = moments::jarque.test(x[!is.na(x)])$p.value)
        x.sum
    }

    col.num <- which(sapply(df, class) == 'numeric')
    lapply(df[, col.num], create_summary)
    
}

summarize_numeric_cols.data.table <- function(df) {
    create_summary <- function(x) {
        x.sum <- summary(x)
        class(x.sum) <- NULL # becomes a named vector
        x.sum <- c(x.sum, 'skewness' = moments::skewness(x, na.rm = TRUE),
                   'kurtosis' = moments::kurtosis(x, na.rm = TRUE),
                   'jarque-brera' = moments::jarque.test(x[!is.na(x)])$p.value)
        x.sum
    }

    col.num <- which(sapply(df, class) == 'numeric')
    lapply(df[, col.num, with = FALSE], create_summary)
}

##########################################################################
# Print multple plots
# Source : http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_%28ggplot2%29/
##########################################################################
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


##########################################################################
# Interactive selection of feature transformations
##########################################################################
# Selects numerical transformation of numerical features interactively
# Saves the transformations in a named vector: 'name' = colname, value = trans
select_num_transf <- function(df, ...) UseMethod('select_num_transf')

select_num_transf.data.frame <- function(df) {
    
    col.num <- which(sapply(df, class) == 'numeric')
    out <- rep('?', length(col.num))
    names(out) <- names(df)[col.num]
    
    x11()
    
    for(i in 1:length(col.num)) {
        x <- df[, col.num[i]]
        x <- x[!is.na(x)]
        x <- x + min(x) # log(a < 0) undefined
        x.plot <- data.frame(x = x)

        # add log
        x.plot <- data.frame(x.plot, logtrans = log(1+x))

        #add Yeo-Johnson
        yj.scaler <- caret::preProcess(x.plot[1], method = c('YeoJohnson'))
        yj.scaled <- predict(yj.scaler, x.plot[1])
        colnames(yj.scaled) <- 'yj'
        x.plot <- cbind(x.plot, yj.scaled)

        # create plots
        p1 <- ggplot(x.plot, aes(x = x)) + geom_line(stat="density") +
            expand_limits(y=0) + ggtitle('Raw')
        p2 <- ggplot(x.plot, aes(x = logtrans)) + geom_line(stat="density") +
            expand_limits(y=0) + ggtitle('log(1+x)')
        p3 <- ggplot(x.plot, aes(x = yj)) + geom_line(stat="density") +
            expand_limits(y=0) + ggtitle('YeoJohnson')

        # plot together
        multiplot(p1, p2, p3, cols = 3)

        # user chooses
        out[i] <- switch(menu(c('rescale', 'log(1+x)', 'YeoJohnson', 'other')),
               'rescale', 'log(1+x)', 'YeoJohnson', 'other')
    }
    dev.off()
    out
}

select_num_transf.data.table <- function(df) {
    
    col.num <- which(sapply(df, class) == 'numeric')
    out <- rep('?', length(col.num))
    names(out) <- copy(names(df))[col.num] # copy since data.table
    
    x11()
    
    for(i in 1:length(col.num)) {
        x <- df[[col.num[i]]]
        x <- x[!is.na(x)]
        x <- x + min(x) # log(a < 0) undefined
        x.plot <- data.frame(x = x)

        # add log
        x.plot <- data.frame(x.plot, logtrans = log(1+x))

        #add Yeo-Johnson
        yj.scaler <- caret::preProcess(x.plot[1], method = c('YeoJohnson'))
        yj.scaled <- predict(yj.scaler, x.plot[1])
        colnames(yj.scaled) <- 'yj'
        x.plot <- cbind(x.plot, yj.scaled)

        # create plots
        p1 <- ggplot(x.plot, aes(x = x)) + geom_line(stat="density") +
            expand_limits(y=0) + ggtitle('Raw')
        p2 <- ggplot(x.plot, aes(x = logtrans)) + geom_line(stat="density") +
            expand_limits(y=0) + ggtitle('log(1+x)')
        p3 <- ggplot(x.plot, aes(x = yj)) + geom_line(stat="density") +
            expand_limits(y=0) + ggtitle('YeoJohnson')

        # plot together
        multiplot(p1, p2, p3, cols = 3)

        # user chooses
        out[i] <- switch(menu(c('rescale', 'log(1+x)', 'YeoJohnson', 'other')),
               'rescale', 'log(1+x)', 'YeoJohnson', 'other')
    }
    dev.off()
    out
}

##########################################################################
# Extraplate single characters and length from string columns
##########################################################################

extrapolate_char <- function(df, ...) UseMethod('extrapolate_char')

extrapolate_char.data.table <- function(df, col) { # columns to process # method for data.table
    
    for(j in 1:length(col)){
        
        name <- names(df)[col[j]]
        cat('Extracting length from', j, 'of', length(col), ', =', name, '\n', sep = ' ')
        
        # time the code
        toprint <- system.time(df[, c(paste('Len_', name, sep = ''))
                       := apply(.SD, 1, function(x) ifelse(is.na(x), 0, nchar(x))),
                       .SDcols = name])
        print(toprint)

                                        # maximum length of string in column col[j]
        local_max <- max(df[, paste('Len_', name, sep = ''), with = FALSE])

                                        # itemize
        cat('Itemizing', j, 'of', length(col), ', =', name, '\n', sep = ' ')
        itnames <- paste('It_', name, '_', 1:local_max, sep = '')

        for(i in 1:local_max) {
            
            cat('Step', i, 'of', local_max, '\n', sep = ' ')
            toprint <- system.time(df[, c(itnames[i]) := apply(.SD, 1, function(x)
                ifelse(is.na(x), '', substr(x, i, i))), .SDcols = name])
            print(toprint)
            
        }
    }
}
                    
extrapolate_char.data.frame <- function(df, col) { # must return a copy
    
    for(j in 1:length(col)){
        
        name <- names(df)[col[j]]
        cat('Extracting length from', j, 'of', length(col), ', =', name, '\n', sep = ' ')
        
        # time the code
        toprint <- system.time(df[c(paste('Len_', name, sep = ''))]
                       <- sapply(df[[name]], function(x) ifelse(is.na(x), 0, nchar(x))))
        print(toprint)

                                        # maximum length of string in column col[j]
        local_max <- max(df[[paste('Len_', name, sep = '')]])

                                        # itemize
        cat('Itemizing', j, 'of', length(col), ', =', name, '\n', sep = ' ')
        itnames <- paste('It_', name, '_', 1:local_max, sep = '')

        for(i in 1:local_max) {
            
            cat('Step', i, 'of', local_max, '\n', sep = ' ')
            toprint <- system.time(df[c(itnames[i])] <- sapply(df[[name]], function(x)
                ifelse(is.na(x), '', substr(x, i, i))))
            print(toprint)
            
        }
    }

    df
}
                    
##########################################################################
# Impact coding http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/
##########################################################################

# return a model of the conditional probability 
# of dependent variable (depvar) by level 
# assumes outcome is logical and not null
impactModel = function(xxcol, ddepvar) { # for an ordered multilevel distribution can be generalized to
# posterior conditional probability
  # if xxcol is factor convert back to character: needed when calling x = c(xcol, xcol)
  if(is.factor(xxcol)) xxcol <- as.character(xxcol)
  indx <- is.na(ddepvar) # to remove NAs from ddepvar
  xcol <- xxcol[!indx]
  depvar <- ddepvar[!indx]
  
  n = length(depvar)
  p = sum(depvar)/n
  
  # duplicate output for NA (average NA towards grand uniform average) 
  x = c(xcol,xcol)
  y = c(depvar, depvar)
  x[(1+n):(2*n)] = NA

  
  levelcounts = table(x, y, useNA="always")

  # This is a smoothed Bayes with 1 previous observation(s!) with positive frequency p
  condprobmodel = (levelcounts[,2]+p)/(levelcounts[,1]+levelcounts[,2]+1.0) 
  # apply model example: applyImpactModel(condprobmodel,data[,varname])
  condprobmodel
}

# apply model to column to essentially return condprobmodel[rawx]
# both NA's and new levels are smoothed to original grand average 
applyImpactModel = function(condprobmodel, xcol) {
    if(is.factor(xcol)) xcol <- as.character(xcol)
    naval = condprobmodel[is.na(names(condprobmodel))] # NA is always created in condprobmodel
    dim = length(xcol)
    condprobvec = numeric(dim) + naval

    N <- length(names(condprobmodel))
    for(i in 1:N) {
        nm <- names(condprobmodel)[i]
        cat('Step', i, 'of', N, '\n', sep = ' ')
        if(!is.na(nm)) {
            toprint <- system.time(condprobvec[xcol==nm] <- condprobmodel[nm])
            print(toprint)
        }
    }
    condprobvec
}

create_impacted <- function(df, ...) UseMethod('create_impacted')

# df = data, tcol = target, fcol = cols to transform; Apply the impacting trick to df
create_impacted.data.table <- function(df, tcol, fcol) {
    nms <- copy(names(df))[fcol]
    N <- length(nms)

    for(i in 1:N) {
        nm = nms[i]
        cat('Impacting', nm, '; Step', i, 'of', N, '\n', sep = ' ')
        condmodel <- impactModel(df[[nm]], df[[tcol]])
        df[, c(paste('Imp_', nm, sep = '')) := applyImpactModel(condmodel, df[[nm]])]
    }
}

# df = data, tcol = target, fcol = cols to transform; Apply the impacting trick to df
create_impacted.data.frame <- function(df, tcol, fcol) {
    nms <- names(df)[fcol]
    N <- length(nms)

    for(i in 1:N) {
        nm = nms[i]
        cat('Impacting', nm, '; Step', i, 'of', N, '\n', sep = ' ')
        condmodel <- impactModel(df[[nm]], df[[tcol]])
        df[[c(paste('Imp_', nm, sep = ''))]] = applyImpactModel(condmodel, df[[nm]])
    }

    df
}

##########################################################################
# Compute sums of individual LETTERS by column
##########################################################################

letters_across <- function(df, ...) UseMethod('letters_across')

letters_across.data.table <- function(df, fcol = NULL) {
    if(is.null(df)) fcol = c(1:ncol(df))

    for(lt in LETTERS) {
        cat('Letter', lt, '\n', sep = ' ')
        df[, paste('Nr_', lt, sep = '') := apply(.SD, 1, function(x) sum(ifelse(is.na(x), 0, x == lt))),
           .SDcols = fcol]
    }

}

# returns in out the modified colums

letters_across.data.frame <- function(df, fcol = NULL) {
    if(is.null(df)) fcol = c(1:ncol(df))

    out <- data.frame(Nr_A = numeric(nrow(df)))
    
    for(lt in LETTERS) {
        cat('Letter', lt, '\n', sep = ' ')
        out[[paste('Nr_', lt, sep = '')]] <- apply(df[,fcol], 1,
                                                   function(x) sum(ifelse(is.na(x), 0, x == lt)))
    }

    out
}

##########################################################################
# Transform the char variables in a given set (fcol = numeric) and those
# in another set extra  to factors;
##########################################################################

char_to_factors <- function(df, ...) UseMethod('char_to_factors')

char_to_factors.data.table <- function(df, fcol = NULL, extra = NULL) { # make generic
    if(is.null(fcol)) {
        fcol <- names(df)[1:ncol(df)]
    } else {
        fcol <- names(df)[fcol]
    }
    
    ch.col <- names(which(df[, sapply(.SD, class), .SDcols = fcol] == 'character'))

    fcol <- intersect(fcol, ch.col)
    if(!is.null(extra)) fcol <- c(fcol, intersect(names(df), extra)) # improve extra both character or numerical

    df[, c(fcol) := lapply(.SD, factor), .SDcols = fcol]
}

char_to_factors.data.frame <- function(df, fcol = NULL, extra = NULL) { # make generic
    if(is.null(fcol)) {
        fcol <- names(df)[1:ncol(df)]
    } else {
        fcol <- names(df)[fcol]
    }
    
    ch.col <- names(which(sapply(df[, fcol], class) == 'character'))

    fcol <- intersect(fcol, ch.col)
    if(!is.null(extra)) fcol <- c(fcol, intersect(names(df), extra)) # improve extra both character or numerical

        for(i in fcol) {
        df[[i]] <- factor(df[[i]])
    }
    df
}

##########################################################################
# convert factors to rfreqs and aggregate factors with similar frequencies
# cols = names of cols; buckets = vector of how many aggregations
##########################################################################
factor_to_freqs <- function(df, ...) UseMethod('factor_to_freqs')

factor_to_freqs.data.table <- function(df, cols, buckets = NULL) {
    N = nrow(df)
    for(icol in c(1:length(cols))) {
        cnm <- cols[icol]
        newnm <- paste('Rfreq_', cnm, sep = '')
        newnm2 <- paste('Ffreq_', cnm, sep = '')

        # create named vector with frequencies
        NAfreq <- sum(is.na(df[[cnm]]))/N
        factab <- table(df[, cols[icol], with = FALSE], useNA = 'always') # useNA to get N.distinct right
        facvec <- as.numeric(factab)/N
        names(facvec) <- rownames(factab)

        # create numeric col
        df[, c(newnm) := facvec[df[[cnm]]]]
        na.indx <- which(is.na(df[[cnm]]))
        df[na.indx, c(newnm) := NAfreq]
        
        # distinct frequencies & bucks
        N.distinct <- length(unique(facvec))
        bucks <- ifelse(is.null(buckets[icol]), N.distinct, buckets[icol])

        # create fac col
        newcol <- cut(df[[newnm]], breaks = bucks)
        levels(newcol) <- paste('F', cnm, c(1:bucks), sep = '_')
        df[, c(newnm2) := newcol]        
        
    }
}

factor_to_freqs.data.frame <- function(df, cols, buckets = NULL) {
    N = nrow(df)
    out <- data.frame(TOREMOVE = rep(0, N))
    for(icol in c(1:length(cols))) {
        cnm <- cols[icol]
        newnm <- paste('Rfreq_', cnm, sep = '')
        newnm2 <- paste('Ffreq_', cnm, sep = '')

        # create named vector with frequencies
        NAfreq <- sum(is.na(df[[cnm]]))/N
        factab <- table(df[[cols[icol]]], useNA = 'always') # useNA to get N.distinct right
        facvec <- as.numeric(factab)/N
        names(facvec) <- rownames(factab)

        # create numeric col
        out[[c(newnm)]] <- facvec[df[[cnm]]]
        na.indx <- which(is.na(df[[cnm]]))
        out[na.indx, c(newnm)] <- NAfreq
        
        # distinct frequencies & bucks
        N.distinct <- length(unique(facvec))
        bucks <- ifelse(is.null(buckets[icol]), N.distinct, buckets[icol])

        # create fac col
        newcol <- cut(df[[newnm]], breaks = bucks)
        levels(newcol) <- paste('F', cnm, c(1:bucks), sep = '_')
        out[[c(newnm2)]] = newcol      
        
    }
    out$TOREMOVE <- NULL
    out
}

##########################################################################
# Recode empty factor "" to a new name
##########################################################################
empty_factor_to_string <- function(df, ...) UseMethod('empty_factor_to_string')

empty_factor_to_string.data.table <- function(df, newname = "EMPTY") {
    for(nm in names(df)) {
        if("" %in% levels(df[[nm]])) {
            p = which(levels(df[[nm]]) == "")
            levels(Alldf[[nm]])[p] = newname
        }
    }
}

empty_factor_to_string.data.frame <- function(df, newname = "EMPTY") {
    for(nm in names(df)) {
        if("" %in% levels(df[[nm]])) {
            p = which(levels(df[[nm]]) == "")
            levels(Alldf[[nm]])[p] = newname
        }
    }

    df
}

##########################################################################
# To write submissions; to modify depending on task
##########################################################################
write_submission <- function(file = file, data = Alldf, index.train = Index_train, pred) {
    write.table(file = file, data.frame(ID = data[!index.train, ID], PredictedProb = pred),
                quote = F, row.names = F, sep = ',')
}
