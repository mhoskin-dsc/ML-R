
run_knn <- function(train.data, train.labels, test.data, test.labels, k.range, 
                    to.console = TRUE,to.file = FALSE, filepath = "") {
  ### Performs n knns and returns a copy of the results.
  ### Runs with a range of different k values as specified by k.range
  ### Inputs:
  ###   train.data - Matrix of m rows and n columns. Each row is an observation 
  ###                for the knn, while each column is a variable
  ###   train.labels - Array of labels for the training data. Should be of dim 
  ###                  1xm, one value for each observation
  ###   test.data - Matrix of p rows and n columns. As with training data, each 
  ###               row is an observation, each column a variable
  ###   test.labels - Array of labels for the test data of dim 1xp. Only used if 
  ###                 to.console is set to TRUE
  ###   k.range - iterable, range of k values to run the knn with.
  ###   to.console - Boolean, whether to print out k values and percentage 
  ###                success to console.
  ###   to.file - Boolean, whether to print to a file or not. If TRUE,  
  ###             file.path must be specified.
  ###   file.path - Str, path of location to write CSV output to. Will write  
  ###               after each knn has been classified.  
  ###
  ### Outputs:
  ###   results.full - Output Matrix of x rows and p columns, where x is the  
  ###                  length of k.range. Each row contains the results for a 
  ###                  specific k value.
  ###                - Each column contains the predicted label for that index of 
  ###                  test data. 

  if (to.file & is.empty(filepath)){
    stop("Unable to write to a file without a filepath specified.")
  }
  
  #build results matrix
  results.full <- matrix(0,nrow = length(k.range), ncol = length(test.labels))

  it <- 1
  for (k in k.range){
    #run KNN and store in matrix. 
    results <- knn(train = train.data, 
                   test = test.data, 
                   cl = train.labels, 
                   k = k)
    
    results.full[it,] <- results
    
    if (to.file){
      write.csv(results.full, file = filepath, append = FALSE)
    }
    
    #print percentage results to console if set
    if(to.console){
      correct <- results == test.labels
      perc <- sum(correct)/length(correct) 
      print(c(k, perc))
    }
    
    it <- it + 1
  }
  
  return(results.full)
}


assess_knn <- function(expected, found){
  ### Compares expected values to calculated values and provides a percentage 
  ### value for each, 
  ###
  ### Inputs:
  ###   expected - 1xp array of correct labels for Test data set.
  ###   found - nxp matrix of predicted labels for Test data set. The number 
  ###           of rows, n, is the number of different models used to loop over.
  ###
  ### Outputs:
  ###   percentages - 
  
  #build results array
  percentages <- array(0, nrow(found))
  
  for (k in 1:nrow(found)){
    #for each row of the found matrix, compare to correct. 
    #'-1' needed as 1 is added to every value in found matrix due to issue with  
    #0 as a classification output
    correct <- expected == found[k,] - 1
    percentages[k] <- sum(correct)/length(correct)
  }
  
  return(percentages)
}