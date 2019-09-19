library(nnet)
# bias,x1,x2 -> first hidden, then second hidden; bias,h1,h2 -> output
# W <- c(-3,10,0,-3,0,10,-2,2,2)
# nn <- nnet(matrix(c(0,0),nrow=1, ncol=2), matrix(c(0), nrow=1, ncol=1), size = 2, maxit = 1)
# # the following line hand-sets the weights
# # nnet can be trained automatically by providing real data above and removing the following line
# #nn$wts <- W
# x <- (0:20) * 0.05
# y <- x
# xy <- expand.grid(x, y)
# z <- predict(nn, xy)
# z <- matrix(z,nrow=21,ncol=21,byrow=TRUE)
# contour(x,y,z)



####raw data
sz <- 50 #units in hidden layer
it <- 50 #maxit
n_train <- 600

training.data <- train$x[1:n_train,] / 255
test.data <- test$x / 255

training.labels <- t(t(train$y[1:n_train]))
training.label <- as.factor(training.labels)

Y <- class.ind(training.label)
Ytest <- class.ind(test$labels)

nn <- nnet(training.data, Y, size = sz, maxit = it, MaxNWts = 50000, softmax = TRUE)

z <- predict(nn, test.data, type='class')

correct <- z == test$labels
perc <- sum(correct) / length(correct)
cat(sz, it, n_train, perc)




mx <- max(train.pca)
mn <- min(train.pca)
test.data <- (test.pca - mn) / (mx - mn)

training.labels <- t(t(train$y[sample.10pc]))
training.label <- as.factor(training.labels)

Y <- class.ind(training.label)


n_pcs_seq <- seq(20,60,5)
it_seq <- seq(70,120,10)
sz <- 60
n_train <- 6000
results.train1.60 <- matrix(0, ncol = length(it_seq), nrow = length(n_pcs_seq))
results.train2.60 <- matrix(0, ncol = length(it_seq), nrow = length(n_pcs_seq))
results.train3.60 <- matrix(0, ncol = length(it_seq), nrow = length(n_pcs_seq))

for (i in c(1:3)){
  for (n_pcs in n_pcs_seq){
    training.data <- (train.pca[sample.10pc,c(1:n_pcs)] - mn) / (mx - mn)
    for (it in it_seq){
      nn <- nnet(training.data, Y, size = sz, maxit = it, MaxNWts = 50000, softmax = TRUE)
      
      z <- predict(nn, test.data, type='class')
      correct <- z == test$y
      perc <- sum(correct) / length(correct)
      
      
      r <- (n_pcs / 5) - 5
      c <- (it / 10) - 10
      if(i == 1){
        results.train1.60[r,c] <- perc
      }
      if(i == 2){
        results.train2.60[r,c] <- perc
      }
      if(i == 3){
        results.train3.60[r,c] <- perc
      }
      cat(it, n_pcs, sz, n_train, perc)
      
    }
  }
}
d <- results.train3.60
cols = c('black','blue','red','orange','green','brown','yellow',"aquamarine","burlywood","darkmagenta")
plot(1:ncol(d), d[1,], type = 'line', ylim = c(0.91, 0.942), main = 'train3.60')
for (i in c(2:nrow(d))){
  lines(1:ncol(d), d[i,],col = cols[i])
}


plot(1:10, percentages.initial10, col = "red",ylim=c(0.85,0.97),main="Comparison of training on initial 10% v sample 10%", 
     xlab="k value for KNN", ylab ="percentage success rate at classifying (%)")
points(1:10, percentages.random10_10, col = "green")


#original pca
sz <- 50 #units in hidden layer
it <- 110 #maxit
n_pcs <- 45
n_train <- 6000

mx <- max(train.pca)
mn <- min(train.pca)

training.data <- (train.pca[sample.10pc,c(1:n_pcs)] - mn) / (mx - mn)
test.data <- (test.pca - mn) / (mx - mn)

training.labels <- t(t(train$y[sample.10pc]))
training.label <- as.factor(training.labels)

Y <- class.ind(training.label)
Ytest <- class.ind(test$labels)

nn <- nnet(training.data, Y, size = sz, maxit = it, MaxNWts = 50000, softmax = TRUE)

z <- predict(nn, test.data, type='class')

correct <- z == test$y
perc <- sum(correct) / length(correct)
cat(sz, it, n_train, perc)

