setwd("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/")

readInData <- function(file){
  data = read.csv(file)
  
  
  m = nrow(data)
  n = ncol(data)
  m
  n
  
  new_names = rep("", n-1)
  for(i in seq(1,n-1,8)){
    new_names[i] = paste("accX",i,sep = "")
  }
  
  for(i in seq(2,n-1,8)){
    new_names[i] = paste("accy",i,sep = "")
  }
  
  for(i in seq(3,n-1,8)){
    new_names[i] = paste("accZ",i,sep = "")
  }
  
  for(i in seq(4,n-1,8)){
    new_names[i] = paste("gyroX",i,sep = "")
  }
  
  for(i in seq(5,n-1,8)){
    new_names[i] = paste("gyroY",i,sep = "")
  }
  
  for(i in seq(6,n-1,8)){
    new_names[i] = paste("gyroZ",i,sep = "")
  }
  
  for(i in seq(7,n-1,8)){
    new_names[i] = paste("lmspeed",i,sep = "")
  }
  
  for(i in seq(8,n-1,8)){
    new_names[i] = paste("rmspeed",i,sep = "")
  }
  
  new_names[n] = "collision"

  
  names(data) <- new_names
  
  
  
  # split = 0.8*m
  # 
  # X = data[1:split,1:(n-1)]
  # y = data[1:split,n]
  # 
  # X_test = data[(split+1):m,1:(n-1)]
  # y_test = data[(split+1):m,n]
  

  X = data[,1:(n-1)]
  Y = data[,n]
  
  newList <- list("X" = X, "Y" = Y)
  
  return(newList)
}

splitTrainTest <- function(X, Y, m){
  
  split = m*nrow(X)
  
  X_train = X[1:split,]
  X_test = X[(split+1):nrow(X),]
  
  y_train = Y[1:split]
  y_test = Y[(split+1):nrow(X)]
  
  newList <- list("X_train" = X_train, "y_train" = y_train, "X_test" = X_test, "y_test" = y_test)
  
  return(newList)
}


train = readInData("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Training/NNtrainingALLNoFrame.csv")

X_train = train$X
y_train = train$Y

test = readInData("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Test/NNtrainingAllNoFrame.csv")
X_test = test$X
y_test = test$Y

# d = readInData("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/NNtrainingALLNoFrame.csv")
# # d = readInData("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/NNtrainingALL.csv")
# 
# d2 = splitTrainTest(d$X, d$Y, 0.7)
# 
# X_train = d2$X_train
# y_train = d2$y_train
# 
# X_test = d2$X_test
# y_test = d2$y_test

sum(y_test)

library(rpart)
tree <- rpart(y_train ~ ., data=X_train, method="class",
              control=rpart.control(cp=0.0))


eval <- function(X_test, y_test, tree){
  
  P <- predict(tree, X_test)
  
  pred_class = rep(0,length(P[,1]))
  
  correct_count = 0

  TP = 0
  TN = 0
  FP = 0
  FN = 0

  for(i in 1:length(P[,1])){

    if(P[i,1] >= 0.5 ){
      pred_class[i] = 0
    }
    else {
      pred_class[i] = 1
    }

    if(pred_class[i] == 1 && y_test[i] == 1){
      TP = TP + 1
    }

    if(pred_class[i] == 1 && y_test[i] == 0){
      FP = FP + 1
    }

    if(pred_class[i] == 0 && y_test[i] == 1){
      FN = FN + 1
    }

    if(pred_class[i] == 0 && y_test[i] == 0){
      TN = TN + 1
    }
  }
  
  # ws = 30
  # 
  # for(i in seq(1,length(P[,1]), ws)){
  #   y_pred = 0
  #   y = 0
  #   for(j in 1:ws){
  #     
  #     if(i +j <= length(P[,1])){
  #       if(P[i+j,1] < 0.5 ){
  #         y_pred = 1
  #       }
  #       
  #       if(y_test[i+j] == 1){
  #         y = 1
  #       }
  #     }
  # 
  #   }
  #   
  #   if(y == 1 && y_pred == 1) TP = TP +1
  #   if(y == 0 && y_pred == 1) FP = FP +1
  #   if(y == 1 && y_pred == 0) FN = FN +1
  #   if(y == 0 && y_pred == 0) TN = TN +1
  # 
  # }
  
  conf = matrix(c(TN,FP,FN,TP), byrow = TRUE, nrow = 2, ncol = 2, dimnames = list(c("keine Kollision","Kollision") ,c("keine Kollision vorhergesagt", "Kollision vorhergesagt")))


  return(conf)
}

evalTree <- function(y_test, X_test, tree, th, toleranceWindow){
  FP = 0
  FN = 0
  TP = 0
  TN = 0
  
  P <- predict(tree, X_test)
  
  pred_class = rep(0,length(P[,1]))
  
  ws = toleranceWindow
  
  for(i in which(y_test == 1)){
    if(length(y_test) > i + ws && i - ws > 0){
      col_det = FALSE
      for(q in c((-ws):(ws))){
        if(P[i + q,1] >= th){
          col_det = TRUE
        }
      }
      
      if(col_det == TRUE) TP = TP + 1
      else FN = FN + 1
    }
  }
  
  lastIndexHigherThanTh = -1000
  for(i in which(P[,1] >= th)){
    if(length(y_test) > i + ws && i - ws > 0 && i - lastIndexHigherThanTh > 100){
      
      lastIndexHigherThanTh = i
      
      actual_col = FALSE
      for(q in c((-ws):(ws))){
        if(y_test[i + q] == 1){
          actual_col = TRUE
        }
      }
      
      if(actual_col == FALSE) FP = FP + 1
      
    }
  }
  
  TN = length(y_test) - FP - TP - FN
  
  
  maxCrit = TN + TP
  
  newList <- list("TP" = TP, "TN" = TN, "FN" = FN, "FP" = FP, "maxCrit" = maxCrit, "colCount" = length(which(y_test == 1)) * 11, "th" = th)
  
  return(newList)
}


conf_matrix = eval(X_test, y_test, tree)
print(conf_matrix)

conf_matrix[1,1] = conf_matrix[1,1]/50

write.table(conf_matrix,file = "TreeConfusionNoFrame.csv", sep = ",", col.names=NA, quote=FALSE)

# conf_matrix = evalTree(y_test, X_test, tree,0.5,100)
# print(conf_matrix)

# conf = matrix(c(conf_matrix$TN,conf_matrix$FP,conf_matrix$FN,conf_matrix$TP), byrow = TRUE, nrow = 2, ncol = 2, dimnames = list(c("keine Kollision","Kollision") ,c("keine Kollision vorhergesagt", "Kollision vorhergesagt")))
# 
# write.table(conf,file = "TreeConfusion.csv", sep = ",", col.names=NA, quote=FALSE)


# install.packages("rpart.plot")

# library(rpart)				        # Popular decision tree algorithm
# library(rattle)					# Fancy tree plot
library(rpart.plot)				# Enhanced tree plots
# library(RColorBrewer)				# Color selection for fancy tree plot
# library(party)					# Alternative decision tree algorithm
# library(partykit)				# Convert rpart object to BinaryTree
# library(caret)					# Just a data source for this script

pdf("treeNoFrame")
rpart.plot(tree)
dev.off()


# citation(package = "rpart", lib.loc = NULL, auto = NULL)
# readCitationFile(file, meta = NULL)
