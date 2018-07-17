setwd("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Test/")

# t1 = read.csv("./s20/confusion_0.29.csv")
# t2 = read.csv("./s21/confusion_0.26.csv")
# t3 = read.csv("./s22/confusion_1.csv")
# t4 = read.csv("./s23/confusion_0.92.csv")
# t5 = read.csv("./s24/confusion_0.3.csv")
# t6 = read.csv("./s25/confusion_0.31.csv")
# t7 = read.csv("./s26/confusion_0.66.csv")
# t8 = read.csv("./s27/confusion_0.77.csv")
# t9 = read.csv("./s28/confusion_0.27.csv")

# t1 = read.csv("./s20/confusion_0.3.csv")
# t2 = read.csv("./s21/confusion_0.3.csv")
# t3 = read.csv("./s22/confusion_0.3.csv")
# t4 = read.csv("./s23/confusion_0.3.csv")
# t5 = read.csv("./s24/confusion_0.3.csv")
# t6 = read.csv("./s25/confusion_0.3.csv")
# t7 = read.csv("./s26/confusion_0.3.csv")
# t8 = read.csv("./s27/confusion_0.3.csv")
# t9 = read.csv("./s28/confusion_0.3.csv")
# 
# t10 = read.csv("./s5/confusion_0.3.csv")
# t11 = read.csv("./s6/confusion_0.3.csv")
# t12 = read.csv("./s7/confusion_0.3.csv")
# t13 = read.csv("./s8/confusion_0.3.csv")
# t14 = read.csv("./s9/confusion_0.3.csv")
# t15 = read.csv("./s10/confusion_0.3.csv")
# t16 = read.csv("./s11/confusion_0.3.csv")
# t17 = read.csv("./s12/confusion_0.3.csv")
# t18 = read.csv("./s13/confusion_0.3.csv")
# 
# t19 = read.csv("./s14/confusion_0.3.csv")
# t20 = read.csv("./s15/confusion_0.3.csv")
# t21 = read.csv("./s16/confusion_0.3.csv")
# t22 = read.csv("./s17/confusion_0.3.csv")
# t23 = read.csv("./s18/confusion_0.3.csv")
# t23 = read.csv("./s19/confusion_0.3.csv")



t1 = read.csv("./t1/confusion_0.3.csv")
t2 = read.csv("./t2/confusion_0.3.csv")
t3 = read.csv("./t3/confusion_0.3.csv")
t5 = read.csv("./t5/confusion_0.3.csv")
t6 = read.csv("./t6/confusion_0.3.csv")
t7 = read.csv("./t7/confusion_0.3.csv")


add_table <- function(table, table_to_add){
  for(i in 1:2){
    for(j in 1:2){
      table[i,j+1] = table[i,j+1] + table_to_add[i,j+1]
    }
  }
  
  return(table)
}

conf_sum = t1

conf_sum = add_table(conf_sum,t2)
conf_sum = add_table(conf_sum,t3)
conf_sum = add_table(conf_sum,t5)
conf_sum = add_table(conf_sum,t6)
conf_sum = add_table(conf_sum,t7)

conf_sum


conf_sum[1,2] = conf_sum[1,2]/50/50

write.table(conf_sum,file = paste("confusion_NN_prediction.csv", sep = ""), sep = ",", col.names=NA, quote=FALSE)


#---------------------------------------------------------------------------------
# Threshold
#---------------------------------------------------------------------------------

norm_vector <- function(x,y,z){
  return(sqrt(x*x + y*y + z*z))
}


Threshold <- function(data, th, toleranceWindow){
  FP = 0
  FN = 0
  TP = 0
  TN = 0
  
  ws = 50
  
  for(i in which(data$collision == 1)){
    if(length(data$collision) > i + ws && i - ws > 0){
      col_det = FALSE
      for(q in c((-ws):(ws))){
        if(norm_vector(data$accX[i + q], data$accY[i + q], data$accZ[i + q]) >= th){
          col_det = TRUE
        }
      }
      
      if(col_det == TRUE) {
        TP = TP + 1
        # offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
        # changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("TP_" ,i, sep = ""))
      }
      else {
        # offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
        # changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("FN_" ,i, sep = ""))
        FN = FN + 1
      }
    }
  }
  
  norms = rep(0,length(data$collision))
  for(i in 1:length(data$collision)){
    norms[i] = norm_vector(data$accX[i], data$accY[i], data$accZ[i])
  }
  

  lastIndexHigherThanTh = -1000
  for(i in which(norms >= th)){
    if(length(data$collision) > i + ws && i - ws > 0 && i - lastIndexHigherThanTh > 100){
      
      lastIndexHigherThanTh = i
      
      actual_col = FALSE
      for(q in c((-ws):(ws))){
        if(data$collision[i + q] == 1){
          actual_col = TRUE
        }
      }
      
      if(actual_col == FALSE){
        FP = FP + 1
        
        # offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
        # changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("FP_" ,i, sep = ""))
      }
      
    }
  }
  
  TN = length(data$collision) - FP - TP - FN
  
  maxCrit = TN + TP
  
  newList <- list("TP" = TP, "TN" = TN, "FN" = FN, "FP" = FP, "maxCrit" = maxCrit, "colCount" = length(which(data$collision == 1)) * 1, "th" = th)
  
  return(newList)
}


predictions_with_threshold <- function(data, th){
  predictions = rep(0,length(data$collision))
  
  for(i in 1:length(data$collision)){
    norm_current = norm_vector(data$accX[i], data$accY[i], data$accZ[i])
    
    if(norm_current >= th) predictions[i] = 1
  }
  
  return(predictions)
}

train_acc_threshold <- function(data){
  
  # for(i in seq(0,1,0.01)){
  #   predictions_with_threshold(data,i)
  # }
  
  m_ = 0
  mOb = 0
  for(i in seq(-40,40.0,1)){
    ob = Threshold(data,i, toleranceWindow = 100)
    
    if(m_ <= ob$maxCrit ){
      m_ = ob$maxCrit
      mOb = ob
    }
  }
  
  # mOb = Threshold(data,0.3, toleranceWindow = 100)
  
  return(mOb)
}

#---------------------------------------------------------------------------------
setwd("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Training/")

data = read.table("Th_training.txt")

if(ncol(data) == 12) {
  colnames(data) <- c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "NNoutput", "collision", "pythonTime")
} else {
  colnames(data) <- c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "collision", "pythonTime")
}

data = na.omit(data)


mOb = train_acc_threshold(data)

conf = matrix(c(mOb$TN,mOb$FP,mOb$FN,mOb$TP), byrow = TRUE, nrow = 2, ncol = 2, dimnames = list(c("keine Kollision","Kollision") ,c("keine Kollision vorhergesagt", "Kollision vorhergesagt")))

write.table(conf,file = paste("confusion_ThresholdTraining.csv", sep = ""), sep = ",", col.names=NA, quote=FALSE)


setwd("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Test/")
data = read.table("Th_test.txt")

if(ncol(data) == 12) {
  colnames(data) <- c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "NNoutput", "collision", "pythonTime")
} else {
  colnames(data) <- c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "collision", "pythonTime")
}

data = na.omit(data)


Ob = Threshold(data,mOb$th, toleranceWindow = 100)

conf = matrix(c(Ob$TN/50,Ob$FP,Ob$FN,Ob$TP), byrow = TRUE, nrow = 2, ncol = 2, dimnames = list(c("keine Kollision","Kollision") ,c("keine Kollision vorhergesagt", "Kollision vorhergesagt")))

write.table(conf,file = paste("confusion_ThresholdTest.csv", sep = ""), sep = ",", col.names=NA, quote=FALSE)



#------------------------------------------------------------------------------------


conf_sum

conf


