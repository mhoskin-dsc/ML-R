#assumes that PCA has been done elsewhere. 
#extended nnet for higher value iterations - about 24 hours

set.seed(11235813)

training.labels <- t(t(train$y))
training.label <- as.factor(training.labels)

Y <- class.ind(training.label)

n_train <- 60000
n_pcs_seq <- seq(50,70,10)
it_seq <- c(120,130,140,150,160,170,180,190,200,250,300,350,400,450,500)
sz_seq <- seq(40,60,10)

results.train <- array(rep(0, length(n_pcs_seq) * length(it_seq) * length(sz_seq)),
                       c(length(n_pcs_seq), length(it_seq), length(sz_seq)))

best_nn_perc <- 0

for (sz in sz_seq){
  for (n_pcs in n_pcs_seq){
    #normalise training data
    mx <- max(train.pca)
    mn <- min(train.pca)
    training.data <- (train.pca[,c(1:n_pcs)] - mn) / (mx - mn)
    test.data <- (test.pca - mn) / (mx - mn)
    
    it_iterator <- 0
    for (it in it_seq){
      it_iterator <- it_iterator + 1
      
      nn <- nnet(training.data, Y, size = sz, maxit = it, 
                 MaxNWts = 50000, softmax = TRUE)
      
      z <- predict(nn, test.data, type='class')
      correct <- z == test$y
      perc <- sum(correct) / length(correct)
      
      r <- (n_pcs / 10) - (n_pcs_seq[1] / 10 - 1)
      c <- it_iterator
      d <- (sz / 10) - (sz_seq[1] / 10 - 1)
      
      results.train[r,c,d] <- perc
      cat(sz, n_pcs, it, n_train, perc, '\n', file = "log.txt", append=TRUE)
      
      
      if (perc > best_nn_perc){
        best_nn_perc <- perc
        cat(sz, n_pcs, it, n_train, perc, '\n', file = "log_best.txt", append=TRUE)
        best_nn <- nn
        best_it <- it
        best_pc <- n_pcs
        best_sz <- sz
      }
      
      
    }
  }
}

plot(it_seq, apply(results.train[,,1], MARGIN = 2, mean), pch = 0, 
     xlab = 'Iterations', ylab='Percentage Success', 
     main = paste('Mean percentage success as nnet iterates',
                  ' with different size hidden layers'), 
     type = 'l', ylim = c(0.925,0.975))
lines(it_seq, apply(results.train[,,2], MARGIN = 2, mean), col = 'red')
lines(it_seq, apply(results.train[,,3], MARGIN = 2, mean), col = 'blue')
legend(x = 'bottomright', legend = c('40','50','60'), col = c('black','red','blue'), 
       bty = 'n', lty = c(1,1,1))

plot(n_pcs_seq, apply(results.train[,,1], MARGIN = 1, mean), pch = 0, 
     xlab = 'n_pcs', ylab='Percentage Success', 
     main = paste('Mean percentage success with different ',
                  'pcs with different size hidden layers'), 
     type = 'l', ylim = c(0.92,0.96))
lines(n_pcs_seq, apply(results.train[,,2], MARGIN = 1, mean), col = 'red')
lines(n_pcs_seq, apply(results.train[,,3], MARGIN = 1, mean), col = 'blue')
legend(x = 'bottomright', legend = c('40','50','60'), col = c('black','red','blue'), 
       bty = 'n', lty = c(1,1,1))

#5:14 chosen as plateau of iteration shelf
plot(n_pcs_seq, apply(results.train[,5:14,1], MARGIN = 1, mean), pch = 0, 
     xlab = 'n_pcs', ylab='Percentage Success', 
     main = paste('Mean percentage success with different ',
                  'pcs with different size hidden layers'), 
     type = 'l', ylim = c(0.92,0.965))
lines(n_pcs_seq, apply(results.train[,5:14,2], MARGIN = 1, mean), col = 'red')
lines(n_pcs_seq, apply(results.train[,5:14,3], MARGIN = 1, mean), col = 'blue')
legend(x = 'bottomright', legend = c('40','50','60'), col = c('black','red','blue'), 
       bty = 'n', lty = c(1,1,1))
