library(class)

for (k in c(1,2,3,4,5,6,7,8,9,10)){ 
  res <- knn(train$x[1:6000,], test$x[1:1000,], train$y[1:6000], k)
  correct <- res == test$y[1:1000]
  print(k)
  print(sum(correct)/length(correct))
}

# [1] 1
# [1] 0.904
# [1] 2
# [1] 0.89
# [1] 3
# [1] 0.913
# [1] 4
# [1] 0.915
# [1] 5
# [1] 0.92
# [1] 6
# [1] 0.91
# [1] 7
# [1] 0.912
# [1] 8
# [1] 0.914
# [1] 9
# [1] 0.914
# [1] 10
# [1] 0.909

for (k in c(1,2,3,4,5,6,7,8,9,10)){ 
  res <- knn(train$x[54001:60000,], test$x[1:1000,], train$y[54001:60000], k)
  correct <- res == test$y[1:1000]
  print(k)
  print(sum(correct)/length(correct))
}

# [1] 1
# [1] 0.919
# [1] 2
# [1] 0.903
# [1] 3
# [1] 0.913
# [1] 4
# [1] 0.903
# [1] 5
# [1] 0.907
# [1] 6
# [1] 0.905
# [1] 7
# [1] 0.906
# [1] 8
# [1] 0.902
# [1] 9
# [1] 0.893
# [1] 10
# [1] 0.894

for (k in c(7)){ 
  res <- knn(train$x[54001:60000,], test$x[1:10000,], train$y[54001:60000], k)
  correct <- res == test$y[1:10000]
  print(k)
  print(sum(correct)/length(correct))
}

# [1] 1
# [1] 0.9361
# [1] 2
# [1] 0.9233
# [1] 3
# [1] 0.9378
# [1] 4
# [1] 0.9349
# [1] 5
# [1] 0.9342
# [1] 6
# [1] 0.932
# [1] 7
# [1] 0.9323

for (k in c(1,2,3,4,5,6,7,8,9,10)){ 
  sample.idx = sample(1:60000, 6000)
  res <- knn(train$x[sample.idx,], test$x[1:1000,], train$y[sample.idx], k)
  correct <- res == test$y[1:1000]
  print(k)
  print(sum(correct)/length(correct))
}

# [1] 1
# [1] 0.917
# [1] 2
# [1] 0.915
# [1] 3
# [1] 0.917
# [1] 4
# [1] 0.917
# [1] 5
# [1] 0.92
# [1] 6
# [1] 0.924
# [1] 7
# [1] 0.906
# [1] 8
# [1] 0.914
# [1] 9
# [1] 0.916
# [1] 10
# [1] 0.913

set.seed(42)
for (k in c(1,2,3,4,5,6,7,8,9,10)){ 
  sample.idx = sample(1:60000, 6000)
  res <- knn(train$x[sample.idx,], test$x[1:1000,], train$y[sample.idx], k)
  correct <- res == test$y[1:1000]
  print(k)
  print(sum(correct)/length(correct))
}

# [1] 1
# [1] 0.92
# [1] 2
# [1] 0.898
# [1] 3
# [1] 0.913
# [1] 4
# [1] 0.918
# [1] 5
# [1] 0.924
# [1] 6
# [1] 0.905
# [1] 7
# [1] 0.902
# [1] 8
# [1] 0.915
# [1] 9
# [1] 0.905
# [1] 10
# [1] 0.902

full.pca <- prcomp(train$x,center = TRUE)
train.pca <- full.pca$x
test.pca <- predict(full.pca, test$x)

set.seed(42)
sample.idx = sample(1:60000, 6000)
pca.to.use <- 1:50
predict <- knn(train = train.pca[sample.idx,pca.to.use], 
               test = test.pca[,pca.to.use],
               cl = train$y[sample.idx],
               k = 3)

correct <- predict == test$y
print(sum(correct)/length(correct))
# [1] 0.9513

upper.bounds <- 2:100
k.range <- 1:10
results.train <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
results.test <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
for (k in k.range){
  # results.train <- c()
  # results.test <- c()
  for (upper.bound in upper.bounds){
    pca.to.use <- 1:upper.bound
    predict.train <- knn(train = train.pca[sample.idx,pca.to.use], 
                   test = train.pca[sample.idx,pca.to.use],
                   cl = train$y[sample.idx],
                   k = k)
    correct <- predict.train == train$y[sample.idx]
    results.train[k,upper.bound - 1] <- (sum(correct)/length(correct))
    # print(sum(correct)/length(correct))
    predict.test <- knn(train = train.pca[sample.idx,pca.to.use], 
                         test = test.pca[,pca.to.use],
                         cl = train$y[sample.idx],
                         k = k)
    correct <- predict.test == test$y
    results.test[k,upper.bound - 1] <- (sum(correct)/length(correct))
  }

  windows()
  plot(upper.bounds, results.train[k,], col = "blue", ylim=c(0,1), main = k)
  points(upper.bounds, results.test[k,], col = "red")
  
  legend("bottomright", legend = c("Training","Test"), col = c("blue", "red"), pch = 1)
}

#99x10 arrays, Cols = #principle components to go up to, Rows = k <- 1:10
# write.csv(results.test, "test_results.csv", row.names = F, col.names = F)
# write.csv(results.train, "train_results.csv", row.names = F, col.names = F)

test_data <- read.csv(file='test_results.csv',header=TRUE)
train_data <- read.csv(file='train_results.csv',header=TRUE)

t <- read.table(file='test_results.csv',header=TRUE)
windows()
matplot(t(test_data), type = 'l')

#plot success rate cut to useful bounding window
for (i in 1:nrow(test_data)){
  windows()
  plot(2:100, test_data[i,], ylim=c(0.9,0.96), main = i)
}

#Find mean values of plateaus of K values from 1-10, from 20 onwards, and from 30 onwards. See minimal difference
rm <- rowMeans(test_data[,20:99])
rm30 <- rowMeans(test_data[,30:99])
rm30 - rm

#plot values. PCs = 30 wins out over averages, 20. 
windows()
plot(1:10, rm, col="blue",ylim=c(0.93,0.96))
points(1:10, rm30, col="red")
points(1:10, test_data[,20], col="green")
points(1:10, test_data[,30], col="chocolate")

#plot values. PCs = 30 wins out over averages, 20. 
windows()
plot(1:10, rm, col="blue",ylim=c(0.93,0.96))
points(1:10, test_data[,25], col="red")
points(1:10, test_data[,35], col="green")
points(1:10, test_data[,30], col="chocolate")
points(1:10, test_data[,40], col="magenta")


k.3 <- test_data[3,]
windows()
plot(29:41, k.3[29:41], ylim=c(0.9,0.96))

#Lets use 31 Principle Components, and k = 3
colnames(k.3[30])
colnames(k.3) <- paste("v", c(2:100), sep="")

library(tictoc)
####bounding area
tic("bounding area")
train.bound <- apply(train$x, 2, max)
test.bound <- apply(test$x, 2, max)
train.zero <- train.bound == 0
test.zero <- test.bound == 0

full.zero <- train.zero & test.zero

test.digit <- train$x[1,]

show_digit(test.digit)
test.digit.cropped <- test.digit
test.digit.cropped[train.zero] <- 255

show_digit(test.digit.cropped)
test.digit.cropped <- test.digit
test.digit.cropped[test.zero] <- 255

show_digit(test.digit.cropped)
test.digit.cropped <- test.digit
test.digit.cropped[full.zero] <- 255
show_digit(test.digit.cropped)

#719 - cuts out the 65 never used cells. 
length(test.digit[!full.zero])

#cut data
full.pca <- prcomp(train$x[, !full.zero],center = TRUE)
train.pca <- full.pca$x
test.pca <- predict(full.pca, test$x[, !full.zero])

set.seed(42)
sample.idx = sample(1:60000, 6000)

upper.bounds <- 2:50
k.range <- seq(1,8)
results.train.cut <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
results.test.cut <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
for (k in k.range){
  # results.train <- c()
  # results.test <- c()
  for (upper.bound in upper.bounds){
    pca.to.use <- 1:upper.bound
    predict.train <- knn(train = train.pca[sample.idx,pca.to.use], 
                         test = train.pca[sample.idx,pca.to.use],
                         cl = train$y[sample.idx],
                         k = k)
    correct <- predict.train == train$y[sample.idx]
    results.train.cut[k,upper.bound - 1] <- (sum(correct)/length(correct))
    # print(sum(correct)/length(correct))
    predict.test <- knn(train = train.pca[sample.idx,pca.to.use], 
                        test = test.pca[,pca.to.use],
                        cl = train$y[sample.idx],
                        k = k)
    correct <- predict.test == test$y
    results.test.cut[k,upper.bound - 1] <- (sum(correct)/length(correct))
  }
  
  windows()
  plot(upper.bounds, results.train.cut[k,], col = "blue", ylim=c(0,1), main = k)
  points(upper.bounds, results.test.cut[k,], col = "red")
  
  legend("bottomright", legend = c("Training","Test"), col = c("blue", "red"), pch = 1)
}
toc()

#original data, no cuts

tic("all")
full.pca <- prcomp(train$x,center = TRUE)
train.pca <- full.pca$x
test.pca <- predict(full.pca, test$x)

set.seed(42)
sample.idx = sample(1:60000, 6000)

upper.bounds <- 2:50
k.range <- seq(1,8)
results.train.default <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
results.test.default <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
for (k in k.range){
  # results.train <- c()
  # results.test <- c()
  for (upper.bound in upper.bounds){
    pca.to.use <- 1:upper.bound
    predict.train <- knn(train = train.pca[sample.idx,pca.to.use], 
                         test = train.pca[sample.idx,pca.to.use],
                         cl = train$y[sample.idx],
                         k = k)
    correct <- predict.train == train$y[sample.idx]
    results.train.default[k,upper.bound - 1] <- (sum(correct)/length(correct))
    # print(sum(correct)/length(correct))
    predict.test <- knn(train = train.pca[sample.idx,pca.to.use], 
                        test = test.pca[,pca.to.use],
                        cl = train$y[sample.idx],
                        k = k)
    correct <- predict.test == test$y
    results.test.default[k,upper.bound - 1] <- (sum(correct)/length(correct))
  }
  
  windows()
  plot(upper.bounds, results.train.default[k,], col = "blue", ylim=c(0,1), main = k)
  points(upper.bounds, results.test.default[k,], col = "red")
  
  legend("bottomright", legend = c("Training","Test"), col = c("blue", "red"), pch = 1)
}
toc()








library(tictoc)



full.pca <- prcomp(train$x,center = TRUE)
train.pca <- full.pca$x
test.pca <- predict(full.pca, test$x)


tic()
upper.bounds <- 2:50
k.range <- seq(3,30,2)
# results.train.default <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
# results.test.default <- matrix(0, ncol = length(upper.bounds), nrow = length(k.range))
for (k in k.range){
  # results.train <- c()
  # results.test <- c()
  for (upper.bound in upper.bounds){
    cat(k,upper.bound)
    pca.to.use <- 1:upper.bound
    predict.train <- knn(train = train.pca[,pca.to.use], 
                         test = train.pca[,pca.to.use],
                         cl = train$y,
                         k = k)
    correct <- predict.train == train$y
    results.train.default[k/2+0.5, upper.bound - 1] <- (sum(correct)/length(correct))
    write.csv(results.train.default, 'train.csv')
    
    # print(sum(correct)/length(correct))
    predict.test <- knn(train = train.pca[,pca.to.use], 
                        test = test.pca[,pca.to.use],
                        cl = train$y,
                        k = k)
    correct <- predict.test == test$y
    results.test.default[k/2+0.5, upper.bound - 1] <- (sum(correct)/length(correct))
    write.csv(results.train.default, 'test.csv')
    
  }
  
  windows()
  plot(upper.bounds, results.train.default[k,], col = "blue", ylim=c(0,1), main = k)
  points(upper.bounds, results.test.default[k,], col = "red")
  
  legend("bottomright", legend = c("Training","Test"), col = c("blue", "red"), pch = 1)
}
toc()
k<-3
for (k in 1:9){#seq(3,17,2)){
  windows()
  plot(upper.bounds, results.train.default[k,], col = "blue", ylim=c(0.8,1), main = k)
  points(upper.bounds, results.test.default[k,], col = "red")

  legend("bottomright", legend = c("Training","Test"), col = c("blue", "red"), pch = 1)
}


col=c("red","green","blue","aquamarine","burlywood","darkmagenta","chartreuse","yellow","chocolate")
windows()
plot(upper.bounds, results.test.default[1,], type = "l", col = col[1], ylim=c(0.9,1), main = k)
for (k in 2:9){#seq(3,17,2)){
  
  #plot(upper.bounds, results.test.default[k,], col = "blue", ylim=c(0.8,1), main = k)
  lines(upper.bounds, results.test.default[k,], col = col[k])
}
#legend("bottomright", legend = c("Training","Test"), col = c("blue", "red"), pch = 1)
