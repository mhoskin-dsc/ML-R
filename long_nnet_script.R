#assumes that PCA has been done elsewhere. 
#Takes about 24 hours to completion

set.seed(11235813)
mx <- max(train.pca)
mn <- min(train.pca)
#can't normalise by dividing by max, as range of values is not 0 - 255, 
#but between a large negative and large positive number. 
#Instead, normalise by (x - min)/(max - min) which will convert to range 0 to 1
test.data <- (test.pca - mn) / (mx - mn) 

training.labels <- t(t(train$y))
training.label <- as.factor(training.labels)

#nnet requires a boolean array of labels
Y <- class.ind(training.label)

n_pcs_seq <- seq(20,60,5)
it_seq <- seq(70,200,10)
sz_seq <- seq(30,50,10)
n_train <- 60000

#initialise output vector
results.train <- array(rep(0, length(n_pcs_seq) * length(it_seq) * length(sz_seq)),
                       c(length(n_pcs_seq), length(it_seq), length(sz_seq)))

for (sz in sz_seq){
  for (n_pcs in n_pcs_seq){
    training.data <- (train.pca[,c(1:n_pcs)] - mn) / (mx - mn)
    for (it in it_seq){
      nn <- nnet(training.data, Y, size = sz, maxit = it, MaxNWts = 50000, softmax = TRUE)
      
      z <- predict(nn, test.data, type='class')
      correct <- z == test$y
      perc <- sum(correct) / length(correct)
      
      
      r <- (n_pcs / 5) - (n_pcs_seq[1] / 5 - 1) 
      c <- (it / 10) - (it_seq[1] / 10 - 1)
      d <- (sz / 10) - (sz_seq[1] / 10 - 1)
      
      results.train[r,c,d] <- perc
      
      cat(sz, n_pcs, it, n_train, perc, '\n', file = "log.txt", append=TRUE)
      
    }
  }
}

plot(it_seq, apply(results.train[,,1], MARGIN = 2, mean), pch = 0, 
     xlab = 'Iterations', ylab='Percentage Success', 
     main = 'Mean percentage success as nnet iterates with different size hidden layers', 
     type = 'l', ylim = c(0.89,0.975))
lines(it_seq, apply(results.train[,,2], MARGIN = 2, mean), col = 'red')
lines(it_seq, apply(results.train[,,3], MARGIN = 2, mean), col = 'blue')
legend(x = 'bottomright', legend = c('30','40','50'), 
       col = c('black','red','blue'), bty = 'n', lty = c(1,1,1))

plot(it_seq, apply(results.train[,,1], MARGIN = 2, max), pch = 0, 
     xlab = 'Iterations', ylab='Percentage Success', 
     main = 'Max percentage success as nnet iterates with different size hidden layers', 
     type = 'l', ylim = c(0.92,0.975))
lines(it_seq, apply(results.train[,,2], MARGIN = 2, max), col = 'red')
lines(it_seq, apply(results.train[,,3], MARGIN = 2, max), col = 'blue')
legend(x = 'bottomright', legend = c('30','40','50'), 
       col = c('black','red','blue'), bty = 'n', lty = c(1,1,1))


plot(n_pcs_seq, apply(results.train1[,,1], MARGIN = 1, mean), pch = 0, 
     xlab = 'n_pcs', ylab='Percentage Success', 
     main = 'Mean percentage success with different pcs with different size hidden layers', 
     type = 'l', ylim = c(0.92,0.96))
lines(n_pcs_seq, apply(results.train[,,2], MARGIN = 1, mean), col = 'red')
lines(n_pcs_seq, apply(results.train[,,3], MARGIN = 1, mean), col = 'blue')
legend(x = 'bottomright', legend = c('30','40','50'), 
       col = c('black','red','blue'), bty = 'n', lty = c(1,1,1))

#5:14 chosen as plateau of iteration shelf
plot(n_pcs_seq, apply(results.train[,5:14,1], MARGIN = 1, mean), pch = 0, 
     xlab = 'n_pcs', ylab='Percentage Success', 
     main = 'Mean percentage success with different pcs with different size hidden layers', 
     type = 'l', ylim = c(0.92,0.965))
lines(n_pcs_seq, apply(results.train[,5:14,2], MARGIN = 1, mean), col = 'red')
lines(n_pcs_seq, apply(results.train[,5:14,3], MARGIN = 1, mean), col = 'blue')
legend(x = 'bottomright', legend = c('30','40','50'), 
       col = c('black','red','blue'), bty = 'n', lty = c(1,1,1))