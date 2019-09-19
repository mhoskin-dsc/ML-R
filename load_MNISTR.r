

# setwd("\\\\NDATA11\\hoskim1$\\My Documents\\R\\Intro-ML-R\\Intro-ML-R")

# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org
load_mnist <- function(train.path = 'data/train-images.idx3-ubyte',
                       test.path = 'data/t10k-images.idx3-ubyte',
                       train.labels.path = 'data/train-labels.idx1-ubyte',
                       test.labels.path = 'data/t10k-labels.idx1-ubyte') {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file(train.path)
  test <<- load_image_file(test.path)
  
  train$y <<- load_label_file(train.labels.path)
  test$y <<- load_label_file(test.labels.path)  
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# load data
# load_mnist()
# library(class)
# 
# 
# 
# 
# 
# 
# library(class)
# for (k in c(4,5,6,7,8,9,10)){ 
#   res <- knn(train$x[1:6000,], test$x[1:1000,], train$y[1:6000], k)
#   correct <- res == test$y[1:1000]
#   print(k)
#   print(sum(correct)/length(correct))
# }
# 
# res.full <- knn(train$x[1:60000,], test$x[1:10000,], train$y[1:60000])
# correct.full <- res.full == test$y[1:10000]
# sum(correct.full)/length(correct.full)
# 
# # inspect contents
# summary(train$x)
# summary(train$y)
# 
# # how the pictures looks like
# train$x[1,]
# show_digit(train$x[1,])
# 
# 
# k=110
# n=10000
# i=404
# iMax=408
# train$x[k:(k+n),i:iMax]
# train$y[k:(k+n)]+1
# pairs(train$x[k:(k+n),i:iMax], col=c("red","green","blue","aquamarine","burlywood","darkmagenta","chartreuse","yellow","chocolate","darkolivegreen")
#       [train$y[k:(k+n)]+1])
# # important: add one to labels (0's seem to make R skip the entry, making the colors inconsistent)
# 
# # plot(train$x[k:(k+n),i], pch=23, 
# #      bg=c("red","green","blue","aquamarine","burlywood","darkmagenta","chartreuse","yellow","chocolate","darkolivegreen")
# #      [train$y[k:(k+n)]])
# # plot(train$x[k:(k+n),i], col=train$y[k:(k+n)])
# # 
# # # having a look at the individual features (pixels)
# # plot(test$x[0:5000,100:101], col=test$y[0:5000])
# # plot(test$x[0:5000,100:101], col=test$y[0:5000])
# # pairs(test$x[5000:5500,100:103], pch=23, col=test$y[5000:5500])
# 
# # some pixels correlate very strongly, other don't
# #C <- cov(train$x)
# #image(C)
# 
# pca.train <- prcomp(train$x,center = TRUE)
# 
# plot(pca.train$sdev)
# pairs(pca.train$x[,1:4],col=c("red","green","blue","aquamarine","burlywood","darkmagenta","chartreuse","yellow","chocolate","darkolivegreen")
#       [train$y+1])
# show_digit(pca.train$center)
# show_digit(pca.train$rotation[,1])
