library(class)
library(tictoc)
library(nnet)
library(rapportools)
wd <- "/Users/training2/Desktop/ML-R"
output <- "./data/"
setwd(wd)

source("load_MNISTR.R")
source("run_knn.R")

### Default paths for files from WD. Uncomment and respecify path to override defaults in load_mnist() 
# train.path <- 'data/train-images.idx3-ubyte'
# test.path <-  'data/t10k-images.idx3-ubyte'
# train.labels.path <-  'data/train-labels.idx1-ubyte'
# test.labels.path <-  'data/t10k-labels.idx1-ubyte'

### Set seed for sample reproducability
set.seed(11235813)

sample.10pc <- sample(1:60000,6000)
sample.test10pc <- sample(1:10000,1000)

### Load in Training, Test, and Label Files.
load_mnist()

######RUN FOR K in 1:10

### Run KNN for first 10% of both training and test data set, for k = 1:10
k.range.initial10 <- c(1:10)
full.results.initial10 <- run_knn(train$x[1:6000,], 
                                      train$y[1:6000], 
                                      test$x[1:1000,], 
                                      test$y[1:1000], 
                                      k.range.initial10,
                                      to.console = FALSE,
                                      to.file = TRUE,
                                      filepath = paste(output,"init10pc10k.csv",sep=""))

percentages.initial10 <- assess_knn(test$y[1:1000], full.results.initial10)


### Run KNN on a random 10% of training, and random 10% of test set, for k = 1:10

k.range.random10 <- c(1:10)
full.results.random10_10 <- run_knn(train$x[sample.10pc,], 
                                        train$y[sample.10pc], 
                                        test$x[sample.test10pc,], 
                                        test$y[sample.test10pc], 
                                        k.range.random10,
                                        to.console = FALSE,
                                        to.file = TRUE,
                                        filepath = paste(output,"rand10pc10k.csv",sep=""))

percentages.random10_10 <- assess_knn(test$y[sample.test10pc], full.results.random10_10)


### Run KNN on a random 10% of training, and the whole test set, for k = 1:10

k.range.random10 <- c(1:10)
full.results.random10_all <- run_knn(train$x[sample.10pc,], 
                                         train$y[sample.10pc], 
                                         test$x, 
                                         test$y, 
                                         k.range.random10,
                                         to.console = FALSE,
                                         to.file = TRUE,
                                         filepath = paste(output,"rand10full10k.csv",sep=""))

percentages.random10_all <- assess_knn(test$y, full.results.random10_all)

### plot first 10% training, 10% test against rand 10% train, rand 10% test, against rand 10%train, all test 
windows()
plot(1:10, percentages.initial10, col = "red",ylim=c(0.85,0.97),main="Comparison of training on initial 10% v sample 10%", 
     xlab="k value for KNN", ylab ="percentage success rate at classifying (%)")
points(1:10, percentages.random10_10, col = "green")
points(1:10, percentages.random10_all, col = "blue")
legend("bottomright", legend = c("f10% v f10%", "r10% v r10%","r10% v all"), col = c("red", "green","blue"), pch = 1)


######RUN FOR K in 1:20

### Run KNN for first 10% of both training and test data set, for k = 1:10
k.range.initial20 <- c(1:20)
full.results.initial10_20k <- run_knn(train$x[1:6000,], 
                                  train$y[1:6000], 
                                  test$x[1:1000,], 
                                  test$y[1:1000], 
                                  k.range.initial20,
                                  to.console = FALSE,
                                  to.file = TRUE,
                                  filepath = paste(output,"init10pc20k.csv",sep=""))

percentages.initial10_20k <- assess_knn(test$y[1:1000], full.results.initial10_20k)


### Run KNN on a random 10% of training, and random 10% of test set, for k = 1:10

k.range.random20 <- c(1:20)
full.results.random10_10_20k <- run_knn(train$x[sample.10pc,], 
                                 train$y[sample.10pc], 
                                 test$x[sample.test10pc,], 
                                 test$y[sample.test10pc], 
                                 k.range.random20,
                                 to.console = FALSE,
                                 to.file = TRUE,
                                 filepath = paste(output,"rand10pc20k.csv",sep=""))

percentages.random10_10_20k <- assess_knn(test$y[sample.test10pc], full.results.random10_10_20k)


### Run KNN on a random 10% of training, and the whole test set, for k = 1:10

k.range.random20 <- c(1:20)
full.results.random10_all_20k <- run_knn(train$x[sample.10pc,], 
                                 train$y[sample.10pc], 
                                 test$x, 
                                 test$y, 
                                 k.range.random20,
                                 to.console = FALSE,
                                 to.file = TRUE,
                                 filepath = paste(output,"rand10pcfull20k.csv",sep=""))

percentages.random10_all_20k <- assess_knn(test$y, full.results.random10_all_20k)

### plot first 10% training, 10% test against rand 10% train, rand 10% test, against rand 10%train, all test 
### success rate starts to drop off after k = 10, so will just test k to 10 in future. 

windows()
plot(1:20, percentages.initial10_20k, col = "red",ylim=c(0.85,0.97),main="Comparison of training on initial 10% v sample 10%", 
     xlab="k value for KNN", ylab ="percentage success rate at classifying (%)")
points(1:20, percentages.random10_10_20k, col = "green")
points(1:20, percentages.random10_all_20k, col = "blue")
legend("bottomright", legend = c("f10% v f10%", "r10% v r10%","r10% v all"), col = c("red", "green","blue"), pch = 1)

conf <- table(test$y,full.results.random10_all_20k[1,]-1)

set.seed(11235813)


### Run PCA on the training data - all of it, apply to test data, first 10%.

full.pca <- prcomp(train$x,center = TRUE)
train.pca <- full.pca$x
test.pca <- predict(full.pca, test$x)
cs <- cumsum(full.pca$sdev^2 / sum(full.pca$sdev^2))
plot(cs)

pr_v <- (full.pca$sdev)^2

prop_v <- pr_v / sum(pr_v)
windows()
plot(prop_v)

### Run KNN on a random 10% of training, and the whole test set, for k = 1:10

upper.bounds.range <- c(2:50)
upper.bounds.length <- length(upper.bounds.range)
k.range <- c(1:10)
test.size <- 1000

data_store <- array(rep(0, length(k.range) * test.size * upper.bounds.length), c(length(k.range), test.size, upper.bounds.length))
matrix_percs <- matrix(0, ncol = upper.bounds.length, nrow = length(k.range))

for (ub in upper.bounds.range){
  
  pca.to.use <- c(1:ub)

  data_store[,,ub-1] <- run_knn(train.pca[sample.10pc,pca.to.use], 
                              train$y[sample.10pc], 
                              test.pca[sample.test10pc,pca.to.use], 
                              test$y[sample.test10pc], 
                              k.range,
                              to.console = FALSE,
                              to.file = FALSE)
  
  matrix_percs[,ub-1] <- assess_knn(test$y[sample.test10pc], data_store[,,ub-1])
}
#max(matrix_percs) = 0.958 # compared with 0.939 from similiar above - ~2% increase





k.range.pca.random10 <- c(1:10)
upper.bounds <- 2:100
results.train <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
results.test <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))

for (upper.bound in upper.bounds){
  pca.to.use <- 1:upper.bound
  full.results.pca.r10 <- run_knn(train.pca[sample.idx, pca.to.use],
                                  train$y[sample.idx],
                                  test.pca[1:1000,pca.to.use],
                                  test$y[1:1000],
                                  k.range.pca.random10)
  results.test[,upper.bound-1] <- full.results.pca.r10
}

  for (upper.bound in upper.bounds){
    pca.to.use <- 1:upper.bound

    predict.test <- knn(train = train.pca[sample.idx,pca.to.use], 
                        test = test.pca[,pca.to.use],
                        cl = train$y[sample.idx],
                        k = k)
    correct <- predict.test == test$y
    results.test[k,upper.bound - 1] <- (sum(correct)/length(correct))
  }
 
#plot first 10 eigenvectors after PCA
par(mfrow = c(2,5), mai = c(0.1,0.1,0.1,0.1))
for (i in c(1:10)){
  show_digit(full.pca$rotation[,i],axes=FALSE,xlab="",ylab="",srt=45)
  text(0.1,0.9, i)
}

#plot first 5, 101:105th, and 601:605th eigenvectors after PCA
par(mfrow = c(3,5), mai = c(0.1,0.1,0.1,0.1))
for (i in c(1:5, 101:105, 780:784)){
  show_digit(full.pca$rotation[,i],axes=FALSE,xlab="",ylab="",srt=45)
  text(0.1,0.9, i)
}

par(mfrow = c(3,5), mai = c(0.1,0.1,0.1,0.1))
combined_result <- array(0, 784)
for (i in c(1:15)){
  combined_result = combined_result + digit * full.pca$rotation[,i]
  show_digit(combined_result)
}







