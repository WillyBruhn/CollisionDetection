#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)!=2) {
  stop("need 1 aruments", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  #args[2] = "out.txt"
}

print("calling R")


wd = args[1]
print(wd)

setwd(wd)

windowSize = as.numeric(args[2])
print(windowSize)


#------------------------------------------------------------------------------

plotColisionWindow <- function(dat,collisionCenter,numOfTimePoints,a,b){
  abline(v = collisionCenter-a, col = "black")
  # abline(v = collisionCenter-numOfTimePoints/4-a, col = "brown")
  # abline(v = collisionCenter+numOfTimePoints/4*3-a, col = "brown")
  
  abline(v = collisionCenter-numOfTimePoints/2.0-a, col = "brown")
  abline(v = collisionCenter+numOfTimePoints/2.0-a, col = "brown")
}

plotDirectionWithActualCollision <- function(data,plotSize, windowSize, a, b){
  
  
  pdf("All")
  par(mfrow=c(3,1))
  
  #par(oma=c(1,1,1,1))
  plot(data$accX[a:b], type = 'l', main = "acc", ylim = plotSize, col = "red")
  plotColisionWindow(data,newCol,windowSize,a,b)
  
  points(data$accY[a:b], type = 'l', main = "y", ylim = plotSize, col = "green")
  plotColisionWindow(data,newCol,windowSize,a,b)
  
  points(data$accZ[a:b], type = 'l', main = "z", ylim = plotSize, col = "blue")
  plotColisionWindow(data,newCol,windowSize,a,b)
  
  
  minG = min(min(data$gyroX[a:b]), min(data$gyroZ[a:b]), min(data$gyroY[a:b]))
  maxG = max(max(data$gyroX[a:b]), max(data$gyroZ[a:b]), max(data$gyroY[a:b]))
  gyroPlotSize = c(minG,maxG)
  
  
  plot(data$gyroX[a:b], type = 'l', main = "gyro", ylim = gyroPlotSize, col = "red")
  plotColisionWindow(data,newCol,windowSize,a,b)
  
  points(data$gyroY[a:b], type = 'l', main = "gy", ylim = gyroPlotSize, col = "green")
  plotColisionWindow(data,newCol,windowSize,a,b)
  
  points(data$gyroZ[a:b], type = 'l', main = "zg", ylim = gyroPlotSize, col = "blue")
  plotColisionWindow(data,newCol,windowSize,a,b)
  
  plot(data$lmspeed[a:b], type = 'l', main = "motor")
  points(data$rmspeed[a:b], type = 'l')
  plotColisionWindow(data,newCol,windowSize,a,b)
  dev.off()
  
}

adjustCollision <- function(data,plotSize,collisionCenter, windowSize, viewSize, offSet){
  coll_tmp = collisionCenter
  coll_tmp = collisionCenter + offSet
  #plotDirectionWithActualCollision(data,plotSize,coll_tmp, windowSize, viewSize)
}

plotWithoutCollision <- function(data, viewSize, a, accLim, gyroLim){
  svg(filename = paste("noCollision", ind, "_", a, ".svg",sep = "") ,width = 8, height = 10.64)
  par(mfrow=c(3,1))
  
  b = a + viewSize
  
  #par(oma=c(1,1,1,1))
  plot(data$accX[a:b], type = 'l', main = "acc", ylim = accLim, col = "red")
  
  points(data$accY[a:b], type = 'l', main = "y", ylim = accLim, col = "green")
  
  points(data$accZ[a:b], type = 'l', main = "z", ylim = accLim, col = "blue")
  
  
  minG = min(min(data$gyroX[a:b]), min(data$gyroZ[a:b]), min(data$gyroY[a:b]))
  maxG = max(max(data$gyroX[a:b]), max(data$gyroZ[a:b]), max(data$gyroY[a:b]))
  gyroPlotSize = c(minG,maxG)
  
  gyroPlotSize = c(-100,100)
  
  
  plot(data$gyroX[a:b], type = 'l', main = "gyro", ylim = gyroLim, col = "red")
  
  points(data$gyroY[a:b], type = 'l', main = "gy", ylim = gyroLim, col = "green")
  
  points(data$gyroZ[a:b], type = 'l', main = "zg", ylim = gyroLim, col = "blue")
  
  plot(data$lmspeed[a:b], type = 'l', main = "motor")
  points(data$rmspeed[a:b], type = 'l')
  
  dev.off()
}


norm_vector <- function(x,y,z){
  return(sqrt(x*x + y*y + z*z))
}



changeCollisionInData <- function(data,oldCol, newCol, windowSize, viewSize, ind, colClass, fName = "nothing", drawColWindow = TRUE){
  
  data$collision[oldCol] = 0
  data$collision[newCol] = colClass

  print(colClass)
  print(newCol)
  
  a = newCol - viewSize/2
  b = newCol + viewSize/2
  
  # pdf(paste("collision_", ind, "_", newCol,sep = ""))
  
  
  if(a > 0 && b < length(data$collision)){
  
    if(fName == "nothing"){
      fName = paste("collision_", ind, "_", newCol, ".svg",sep = "")
    }
    svg(filename =  fName,width = 8, height = 10.64)
    par(mfrow=c(4,1))
    
    #par(oma=c(1,1,1,1))
    plot(data$accX[a:b], type = 'l', main = "Beschleunigung", ylim = plotSize, col = "red", ylab = "m/s²", xlab = "8 ms")
    # plotColisionWindow(data,newCol,windowSize,a,b)
    
    points(-data$accY[a:b], type = 'l', main = "y", ylim = plotSize, col = "green")
    # plotColisionWindow(data,newCol,windowSize,a,b)
    
    points(data$accZ[a:b], type = 'l', main = "z", ylim = plotSize, col = "blue")
    
    if(drawColWindow){
      plotColisionWindow(data,newCol,windowSize,a,b)
    }

    
    norm_current = rep(0,b-a)
    for(i in 1:length(norm_current)){
      norm_current[i] = norm_vector(data$accX[i-1+a], data$accY[i-1+a], data$accZ[i-1+a])
    }

    points(norm_current[1:(b-a)], type = 'l', main = "y", ylim = plotSize, col = "red", lty = 2)
    
    abline(h=8, col = "cyan")
    
    legend(85, 15, legend=c("X-Richtung", "Y-Richtung", "Z-Richtung", "Norm", "Schwellwert"),
           col=c("red","green", "blue", "red", "cyan"), lty = c(1,1,1,2,1), cex=1.0)
    
    
    minG = min(min(data$gyroX[a:b]), min(data$gyroZ[a:b]), min(data$gyroY[a:b]))
    maxG = max(max(data$gyroX[a:b]), max(data$gyroZ[a:b]), max(data$gyroY[a:b]))
    gyroPlotSize = c(minG,maxG)
    
    gyroPlotSize = c(-150,80)
    
    
    plot(data$gyroX[a:b], type = 'l', main = "Gyroskop-Daten", ylim = gyroPlotSize, col = "red", ylab = "m/s" , xlab = "8 ms")
    # plotColisionWindow(data,newCol,windowSize,a,b)
    
    points(data$gyroY[a:b], type = 'l', main = "gy", ylim = gyroPlotSize, col = "green")
    # plotColisionWindow(data,newCol,windowSize,a,b)
    
    points(data$gyroZ[a:b], type = 'l', main = "zg", ylim = gyroPlotSize, col = "blue")
    # plotColisionWindow(data,newCol,windowSize,a,b)
    
    if(drawColWindow){
      plotColisionWindow(data,newCol,windowSize,a,b)
    }
    
    
    legend(85, -50, legend=c("X-Richtung", "Y-Richtung", "Z-Richtung"),
           col=c("red","green", "blue"), lty = c(1,1,1), cex=1.0)
    
    
    
    plot(data$lmspeed[a:b], type = 'l', col = "blue", main = "Motor-Geschwindigkeiten", ylim = c(-250,250), xlab = "8 ms", ylab = "Geschwindigkeits-Stufen")
    points(data$rmspeed[a:b], type = 'l', col = "red")
    # plotColisionWindow(data,newCol,windowSize,a,b)
    
    if(drawColWindow){
      plotColisionWindow(data,newCol,windowSize,a,b)
    }
    
    
    legend(85, -100, legend=c("Links", "Rechts"),
           col=c("blue","red"), lty = c(1,1), cex=1.0)
    
    if(ncol(data) == 12){
      plot(data$NNoutput[a:b], type = 'l', main = "Ausgabe des Neuronalen Netzes", ylim = c(0,1), col = "blue",  xlab = "8 ms", ylab = "Wahrscheinlichkeit für eine Kollision")
      # plotColisionWindow(data,newCol,windowSize,a,b)
      abline(h = 0.3, col = "red")
      
      if(drawColWindow){
        plotColisionWindow(data,newCol,windowSize,a,b)
      }
      
      
      legend(80, 0.9, legend=c("NN", "Schwellwert"),
             col=c("red", "blue"), lty = c(1,1), cex=1.0)
    }
  
    
    dev.off()
    # file.size(paste("collision_", ind, "_", newCol, ".svg",sep = ""))/1000
  }
  
  return(data)
}

# frameSize = in this area collisions are also accepted
generateNNTrainingsData <- function(data, windowSize, numOfVars, overlap, frameSize = c(-4:1)) {
  # Design-matrix
  X <- matrix(ncol=(windowSize*numOfVars), nrow=(length(data$collision)-windowSize))
  Y <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  Y_noFrame <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  normalizeConstMotor = 250
  normalizeConstAcc = 30
  
  normalizeConstGyro = 200
  
  colCount = 0
  colCount_noFrame = 0
  
  x <- rep(0,windowSize*numOfVars) # x,y,z, lmspeed, rmspeed
  y = 0
  y_noFrame = 0
  
  for(i in seq(1,length(data$collision)-windowSize, overlap)){
    #print(paste(i , " / ", length(data$collision)))
    
    y = 0
    y_noFrame = 0
    
    for(j in 1:windowSize){
      # -1 because j starts with 1
      ind = j + (j-1)*(numOfVars-1)
      x[ind] = data$accX[i+j-1]/normalizeConstAcc
      x[ind+1] = data$accY[i+j-1]/normalizeConstAcc
      x[ind+2] = data$accZ[i+j-1]/normalizeConstAcc
      
      
      if(numOfVars == 8){
        x[ind+3] = data$gyroX[i+j-1]/normalizeConstGyro
        x[ind+4] = data$gyroY[i+j-1]/normalizeConstGyro
        x[ind+5] = data$gyroZ[i+j-1]/normalizeConstGyro
        
        x[ind+6] = data$lmspeed[i+j-1]/normalizeConstMotor
        x[ind+7] = data$rmspeed[i+j-1]/normalizeConstMotor
      }
      #   ind = j + (j-1)*(numOfVars-1)
      
      #print(ind)
    }
    
    for(q in frameSize){
      for(t in c(1:20)){
         if(data$collision[i+q+windowSize/2.0] == t){
        # if(data$collision[i+q+windowSize/2.0] == 1){
          colCount = colCount + 1
          y = t
         }
      }
    }
    
    for(t in c(1:20)){
      if(data$collision[i+windowSize/2.0] == t){
        # if(data$collision[i+q+windowSize/2.0] == 1){
        colCount_noFrame = colCount_noFrame + 1
        y_noFrame = t
      }
    }
    
    X[i,] = x
    Y[i,] = y
    
    Y_noFrame[i,] = y_noFrame
  }
  
  # X = data.frame(X)
  # Y = data.frame(Y)
  # 
  # Y_noFrame = data.frame(Y_noFrame)
  
  print(paste(colCount_noFrame, " collisions_no Frame"))
  print(paste(colCount, " collisions"))
  
  newList <- list("X" = X, "Y" = Y, "Y_noFrame" = Y_noFrame)
  
  return(newList)
}

generateNNTrainingsDataOneClass <- function(data, windowSize, numOfVars, overlap, frameSize = c(-4:1)) {
  # Design-matrix
  X <- matrix(ncol=(windowSize*numOfVars), nrow=(length(data$collision)-windowSize))
  Y <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  Y_noFrame <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  normalizeConstMotor = 250
  normalizeConstAcc = 30
  
  normalizeConstGyro = 200
  
  colCount = 0
  colCount_noFrame = 0
  
  x <- rep(0,windowSize*numOfVars) # x,y,z, lmspeed, rmspeed
  y = 0
  y_noFrame = 0
  
  for(i in seq(1,length(data$collision)-windowSize, overlap)){
    #print(paste(i , " / ", length(data$collision)))
    
    y = 0
    y_noFrame = 0
    
    for(j in 1:windowSize){
      # -1 because j starts with 1
      ind = j + (j-1)*(numOfVars-1)
      x[ind] = data$accX[i+j-1]/normalizeConstAcc
      x[ind+1] = data$accY[i+j-1]/normalizeConstAcc
      x[ind+2] = data$accZ[i+j-1]/normalizeConstAcc
      
      
      if(numOfVars == 8){
        x[ind+3] = data$gyroX[i+j-1]/normalizeConstGyro
        x[ind+4] = data$gyroY[i+j-1]/normalizeConstGyro
        x[ind+5] = data$gyroZ[i+j-1]/normalizeConstGyro
        
        x[ind+6] = data$lmspeed[i+j-1]/normalizeConstMotor
        x[ind+7] = data$rmspeed[i+j-1]/normalizeConstMotor
      }
      #   ind = j + (j-1)*(numOfVars-1)
      
      #print(ind)
    }
    
    for(q in frameSize){
      if(data$collision[i+q+windowSize/2.0] == 1){
        # if(data$collision[i+q+windowSize/2.0] == 1){
        colCount = colCount + 1
        y = 1 - abs(q)*2/windowSize
      }
    }
    
    if(data$collision[i+windowSize/2.0] == 1){
      # if(data$collision[i+q+windowSize/2.0] == 1){
      colCount_noFrame = colCount_noFrame + 1
      y_noFrame = 1
    }
    
    X[i,] = x
    Y[i,] = y
    
    Y_noFrame[i,] = y_noFrame
  }
  
  # X = data.frame(X)
  # Y = data.frame(Y)
  # 
  # Y_noFrame = data.frame(Y_noFrame)
  
  print(paste(colCount_noFrame, " collisions_no Frame"))
  print(paste(colCount, " collisions"))
  
  newList <- list("X" = X, "Y" = Y, "Y_noFrame" = Y_noFrame)
  
  return(newList)
}

createFeatures <- function(V){
  ret <- rep(0,1)
  
  ret[1] = max(V)
  ret[2] = min(V)
  ret[3] = mean(V)

  ret[4] = median(V)
  ret[5] = median(V) - mean(V)
  ret[6] = (median(V)-mean(V))^2
  ret[7] = sd(V)/10
  ret[8] = sum(V)/40
  ret[9] = sum(abs(V))/40
  
  
  # x[windowSize*numOfVars + 3] = mean(V)
  # x[windowSize*numOfVars + 4] = mean(V)*mean(V)
  # x[windowSize*numOfVars + 5] = median(V)
  # x[windowSize*numOfVars + 6] = median(V)-mean(V)
  # x[windowSize*numOfVars + 7] = (median(V)-mean(V))^2
  # x[windowSize*numOfVars + 8] = sum(V)
  # x[windowSize*numOfVars + 9] = sd(V)
  
  
  return(ret)
}

generateNNTrainingsDataWithFeatures2 <- function(data, windowSize, numOfVars, overlap, frameSize = c(-4:1)) {
  # Design-matrix
  
  numOfFeatures = length(createFeatures(data$accX[1:(1+windowSize-1)]))*numOfVars
  
  X <- matrix(ncol=(numOfFeatures), nrow=(length(data$collision)-windowSize))
  Y <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  Y_noFrame <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  normalizeConstMotor = 250
  normalizeConstAcc = 30
  
  normalizeConstGyro = 200
  
  colCount = 0
  colCount_noFrame = 0
  
  x <- rep(0,numOfFeatures) # x,y,z, lmspeed, rmspeed
  y = 0
  y_noFrame = 0
  
  
  
  data$accX = data$accX/normalizeConstAcc
  data$accY = data$accY/normalizeConstAcc
  data$accZ = data$accZ/normalizeConstAcc
  
  data$gyroX = data$gyroX/normalizeConstGyro
  data$gyroY = data$gyroY/normalizeConstGyro
  data$gyroZ = data$gyroZ/normalizeConstGyro
  
  data$lmspeed = data$lmspeed/normalizeConstMotor
  data$rmspeed = data$rmspeed/normalizeConstMotor
  
  for(i in seq(1,length(data$collision)-windowSize, overlap)){
    #print(paste(i , " / ", length(data$collision)))
    
    y = 0
    y_noFrame = 0
    
    x_ind = 1
    f = createFeatures(data$accX[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$accY[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$accZ[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$gyroX[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$gyroY[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$gyroZ[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$lmspeed[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    f = createFeatures(data$rmspeed[i:(i+windowSize-1)])
    for(u in 1:length(f)){
      x[x_ind] = f[u]
      
      x_ind = x_ind +1
    }
    
    for(q in frameSize){
      for(t in c(1:20)){
        if(data$collision[i+q+windowSize/2.0] == t){
          # if(data$collision[i+q+windowSize/2.0] == 1){
          colCount = colCount + 1
          y = t
        }
      }
    }
    
    for(t in c(1:20)){
      if(data$collision[i+windowSize/2.0] == t){
        # if(data$collision[i+q+windowSize/2.0] == 1){
        colCount_noFrame = colCount_noFrame + 1
        y_noFrame = t
      }
    }
    
    X[i,] = x
    Y[i,] = y
    
    Y_noFrame[i,] = y_noFrame
  }
  
  # X = data.frame(X)
  # Y = data.frame(Y)
  # 
  # Y_noFrame = data.frame(Y_noFrame)
  
  print(paste(colCount_noFrame, " collisions_no Frame"))
  print(paste(colCount, " collisions"))
  
  newList <- list("X" = X, "Y" = Y, "Y_noFrame" = Y_noFrame)
  
  return(newList)
}

generateNNTrainingsDataWithExtraFeatures <- function(data, windowSize, numOfVars, overlap){
  # Design-matrix
  X
  Y
  
  X_ <- matrix(ncol=(windowSize*numOfVars + 3*9), nrow=(length(data$collision)-windowSize))
  Y_ <- matrix(ncol=1, nrow=(length(data$collision)-windowSize))
  
  normalizeConstMotor = 250
  normalizeConstAcc = 30
  
  colCount = 0
  for(i in seq(1,length(data$collision)-windowSize, overlap)){
    print(paste(i , " / ", length(data$collision)))
    
    x <- rep(0,windowSize*numOfVars + 3*9) # x,y,z, lmspeed, rmspeed + min, max, average, avg ^2, med ,med-avg,( med-avg) ^2, sum, sd
    y = 0
    
    for(j in 1:windowSize){
      # -1 because j starts with 1
      ind = j + (j-1)*(numOfVars-1)
      x[ind] = data$accX[i+j-1]/normalizeConstAcc
      x[ind+1] = data$accY[i+j-1]/normalizeConstAcc
      x[ind+2] = data$accZ[i+j-1]/normalizeConstAcc
      
      if(numOfVars == 5){
        x[ind+3] = data$lmspeed[i+j-1]/normalizeConstMotor
        x[ind+4] = data$rmspeed[i+j-1]/normalizeConstMotor
      }
      #   ind = j + (j-1)*(numOfVars-1)
      
      #print(ind)
    }
    
    if(data$collision[i+windowSize/2.0] == 1){
      colCount = colCount + 1
      y = 1
    }
    
    vals = c(i:(i+windowSize))
    x[windowSize*numOfVars + 1] = min(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 2] = max(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 3] = mean(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 4] = mean(data$accX[vals]/normalizeConstAcc)*mean(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 5] = median(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 6] = median(data$accX[vals]/normalizeConstAcc)-mean(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 7] = (median(data$accX[vals]/normalizeConstAcc)-mean(data$accX[vals]/normalizeConstAcc))^2
    x[windowSize*numOfVars + 8] = sum(data$accX[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 9] = sd(data$accX[vals]/normalizeConstAcc)
    
    x[windowSize*numOfVars + 10] = min(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 11] = max(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 12] = mean(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 13] = mean(data$accY[vals]/normalizeConstAcc)*mean(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 14] = median(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 15] = median(data$accY[vals]/normalizeConstAcc)-mean(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 16] = (median(data$accY[vals]/normalizeConstAcc)-mean(data$accY[vals]/normalizeConstAcc))^2
    x[windowSize*numOfVars + 17] = sum(data$accY[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 18] = sd(data$accY[vals]/normalizeConstAcc)
    
    x[windowSize*numOfVars + 19] = min(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 20] = max(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 21] = mean(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 22] = mean(data$accZ[vals]/normalizeConstAcc)*mean(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 23] = median(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 24] = median(data$accZ[vals]/normalizeConstAcc)-mean(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 25] = (median(data$accZ[vals]/normalizeConstAcc)-mean(data$accZ[vals]/normalizeConstAcc))^2
    x[windowSize*numOfVars + 26] = sum(data$accZ[vals]/normalizeConstAcc)
    x[windowSize*numOfVars + 27] = sd(data$accZ[vals]/normalizeConstAcc)
    
    #X = rbind(X,x)
    #Y = rbind(Y,y)
    
    X_[i,] = x
    Y_[i,] = y
  }
  
  X <- data.frame(X_)
  Y <- data.frame(Y_)
  
  
  print(paste(colCount, " collisions"))
  
  newList <- list("X" = X, "Y" = Y)
  
  return(newList)
}

removeCollision <- function(data, col){
  data$collision[col] = 0
  
  return(data)
}

predictOffSet <- function(data, collisionCenter, viewSize, accFlag){
  a = collisionCenter - viewSize/2.0
  b = collisionCenter + viewSize/2.0
  
  if(b < length(data$collision)){
    if(accFlag == 1){
      return(which.max(abs(data$accY[a:b]))-viewSize/2.0)
    }
    
    gx = which.max(abs(data$accX[a:b]))
    gy = which.max(abs(data$accY[a:b]))
    gz = which.max(abs(data$accZ[a:b]))
    
    gx_max = abs(data$accX[a+gx-1])
    gy_max = abs(data$accY[a+gy-1])
    gz_max = abs(data$accZ[a+gz-1])
    
    print(gx)
    if(gx_max != max(abs(data$accX[a:b])) || gz_max != max(abs(data$accZ[a:b])) || gy_max != max(abs(data$accY[a:b]))) {
      print(paste("error: ", gx_max, " !=? ", max(abs(data$accX[a:b]))))
      print(paste("error: ", gy_max, " !=? ", max(abs(data$accY[a:b]))))
      print(paste("error: ", gz_max, " !=? ", max(abs(data$accZ[a:b]))))
    }
    
    max_ind = gx
    m = gx_max
    
    if(gy_max > m){
      max_ind = gy
      m = gy_max
    }
    
    if(gz_max > m){
      max_ind = gz
      m = gz_max
    }
    
    return(max_ind-viewSize/2.0)
  }
  

  return(-999)
  
}

plotNN <- function(data){
  
  colCenters = which(data$collision == 1)
  
  NN_thresholds = rep(0,length(colCenters))
  j = 1
  
  for(i in colCenters){
    if(i > 100 && i < length(data$collision) -100){
      NN_thresholds[j] = max(data$NNoutput[c((i-100):(i+100))])
      j = j+1
    }
  }
  
  No_collisions = which(data$collision == 0)
  
  pdf("NNoutput")
  plot(data$NNoutput, main = paste("NNoutput avg during col: ", mean(NN_thresholds),sep = ""))
  for(i in colCenters){
    abline(v = i, col = "red")
  }
  dev.off()
  
  for(i in c(1:length(data$collision))){
    
  }
  
  return(NN_thresholds)
}

NNWithThreshold <- function(data, th, toleranceWindow){
  FP = 0
  FN = 0
  TP = 0
  TN = 0
  
  # 
  # for(i in c(200:(length(data$collision)-200))){
  #   NNpred = FALSE
  #   NNactual = FALSE
  #   
  #   for(q in c(-5:5)){
  #     if(data$NNoutput[i + q + 10] >= th){
  #       NNpred = TRUE
  #     }
  #     
  #     if(data$collision[i + q] == 1){
  #       NNactual = TRUE
  #     }
  #   }
  #   
  #   if(NNpred == TRUE && NNactual == TRUE) TP = TP + 1
  #   if(NNpred == FALSE && NNactual == TRUE) FN = FN + 1
  #   if(NNpred == TRUE && NNactual == FALSE) FP = FP + 1
  #   if(NNpred == FALSE && NNactual == FALSE) TN = TN + 1
  # }
  
  ws = 50

  for(i in which(data$collision == 1)){
    if(length(data$collision) > i + ws && i - ws > 0){
      col_det = FALSE
      for(q in c((-ws):(ws))){
        if(data$NNoutput[i + q] >= th){
          col_det = TRUE
        }
      }
      
      if(col_det == TRUE) {
          TP = TP + 1
          offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
          changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("TP_" ,i, sep = ""))
        }
      else {
        offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
        changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("FN_" ,i, sep = ""))
        FN = FN + 1
      }
    }
  }
  
  lastIndexHigherThanTh = -1000
  for(i in which(data$NNoutput >= th)){
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
      
        offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
         changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("FP_" ,i, sep = ""), FALSE)
      }
    
    }
  }
  
  TN = length(data$collision) - FP - TP - FN
  
  
  maxCrit = TN + TP
  
  newList <- list("TP" = TP, "TN" = TN, "FN" = FN, "FP" = FP, "maxCrit" = maxCrit, "colCount" = length(which(data$collision == 1)) * 11, "th" = th)
  
  return(newList)
}

eval <- function(y_test, y_prediction, th, toleranceWindow){
  
  TP = 0
  TN = 0
  FP = 0
  FN = 0

  ws = toleranceWindow
  
  t = 0
  for(i in seq(1,length(y_test), ws)){
    y_pred = 0
    y = 0
    for(j in 1:ws){
      
      if(i +j <= length(y_test)){
        if(y_prediction[i+j] >= th ){
          y_pred = 1
        }
        
        if(y_test[i+j] == 1){
          y = 1
        }
      }
      
    }
    
    if(y == 1 && y_pred == 1) TP = TP +1
    if(y == 0 && y_pred == 1) FP = FP +1
    if(y == 1 && y_pred == 0) FN = FN +1
    if(y == 0 && y_pred == 0) TN = TN +1
    
    t = t +1
    
  }
  
  maxCrit = TN + TP
  
  newList <- list("TP" = TP, "TN" = TN, "FN" = FN, "FP" = FP, "maxCrit" = maxCrit, "total" = t, "th" = th)
  
  return(newList)
}



#--------------------------------------------------------------------------------------------------------------------------------------------------

setwd("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Test/t7")

#str1 = "/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/TrainingMultiClass/s4"

# str1 = sub(".*/", "", wd)
# str1 = sub("[0-9]+", "", str1)
# str1


colClass = 1

# b = 1,s = 2, f = 3, t = 4
# b = backwards collision
# s = forward collision
# f collision instant forward
# t collision instant backward

# if(str1 == "s"){
#   colClass = 2
# } else if(str1 == "f"){
#   colClass = 3
# } else if(str1 == "t"){
#   colClass = 4
# }


data = read.table("output.txt")

#col.names = c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "NNoutput", "collision", "pythonTime"), fill = TRUE)

if(ncol(data) == 12) {
  colnames(data) <- c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "NNoutput", "collision", "pythonTime")
} else {
  colnames(data) <- c("arduinoTime", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "lmspeed", "rmspeed", "collision", "pythonTime")
}

data = na.omit(data)



plotSize = c(min(data$accX,data$accY, data$accZ), max(data$accX,data$accY, data$accZ))

collisionCenters = which(data$collision == 1)
length(collisionCenters)


# remove the last few entries
if(collisionCenters[length(collisionCenters)] > length(data$collision) - 50){
  data = head(data,-100)
}

collisionCenters = which(data$collision == 1)
length(collisionCenters)


windowSize = 20
viewSize = 100

#plotDirectionWithActualCollision(data,plotSize,collisionCenter, windowSize, viewSize)


#ind = 1
#--------------------------------------------------------------------------------------------------------------------------------------------------
# rerun this pasage over and over again
#--------------------------------------------------------------------------------------------------------------------------------------------------
# ind = 1
# print(paste("collision ", ind, " / ", length(collisionCenters)))
# offSet = predictOffSet(data,collisionCenters[ind],viewSize = viewSize,2)
# adjustCollision(data,plotSize = plotSize,collisionCenter = collisionCenters[ind], windowSize = windowSize, viewSize = viewSize, offSet = offSet)
# 
# #removeCollision(data,collisionCenters[ind])
# 
# print("now adjusting data ...")
# print("are you sure man?")
# changeCollisionInData(data,collisionCenters[ind],collisionCenters[ind]+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind)
# ind = ind +1



# #--------------------------------------------------------------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------------------------------------------------------------

#collisionCenters = c(100,400,600,800)

# par(mfrow=c(2,1))
# pdf("col")
# 
# m = which.max(data$accY)
# 
# 
# m = m -10
# 
# plot(data$accY[(m-5):(m+15)], ylim = c(-5,25))
# 
# 
# dev.off()
#plot(data$accY)


oldInd = 1
if(length(collisionCenters) != 0){
  for(ind in 1:length(collisionCenters)){
    if(oldInd == ind || collisionCenters[oldInd] + windowSize <= collisionCenters[ind]){
      print(paste(collisionCenters[oldInd], " != ", collisionCenters[ind]))
      print(paste("collision ", ind, " / ", length(collisionCenters)))
      offSet = predictOffSet(data,collisionCenters[ind],viewSize = viewSize/2.0, 2)
      
      if(offSet == -999){
        data = removeCollision(data,collisionCenters[ind])
      }
      else {
        adjustCollision(data,plotSize = plotSize,collisionCenter = collisionCenters[ind], windowSize = windowSize, viewSize = viewSize, offSet = offSet)
        
        data = changeCollisionInData(data,collisionCenters[ind],collisionCenters[ind]+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass)
      }

      
    }
   else{
   data = removeCollision(data,collisionCenters[ind])
   }
   oldInd = ind
  }
}

#----------------------------------------------------------------------------------
# this was good
#----------------------------------------------------------------------------------
collisionCenters = which(data$collision == 1)
length(collisionCenters)

obj = generateNNTrainingsData(data,windowSize = windowSize, numOfVars = 8, 1, c(-10:10))
D = cbind(obj$X,obj$Y)
write.table(D, file = "NNtraining.csv", sep=",", col.names = T, row.names = F)

C = cbind(obj$X,obj$Y_noFrame)
write.table(C, file = "NNtrainingNoFrame.csv", sep=",", col.names = T, row.names = F)
#----------------------------------------------------------------------------------
# this was good
#----------------------------------------------------------------------------------


# obj = generateNNTrainingsDataOneClass(data,windowSize = windowSize, numOfVars = 8, 1, c((-windowSize/2.0):(windowSize/2.0)))
# D = cbind(obj$X,obj$Y)
# write.table(D, file = "NNtrainingBlurredClass.csv", sep=",", col.names = T, row.names = F)

# p = obj$X[which(obj$Y_noFrame == colClass),]
# q = colMeans(p, na.rm=TRUE)
# 
# q_acc_x = q[seq(1,length(q),8)]
# q_acc_y = q[seq(2,length(q),8)]
# q_acc_z = q[seq(3,length(q),8)]
# 
# q_g_x = q[seq(4,length(q),8)]
# q_g_y = q[seq(5,length(q),8)]
# q_g_z = q[seq(6,length(q),8)]
# 
# p2 = obj$X[which(obj$Y_noFrame != colClass),]
# 
# q2 = colMeans(p2, na.rm=TRUE)
# 
# q2_acc_x = q2[seq(1,length(q),8)]
# q2_acc_y = q2[seq(2,length(q),8)]
# q2_acc_z = q2[seq(3,length(q),8)]
# 
# q2_g_x = q2[seq(4,length(q),8)]
# q2_g_y = q2[seq(5,length(q),8)]
# q2_g_z = q2[seq(6,length(q),8)]
# 
# 
# 
# 
# pdf("ColVsNoCol.pdf")
# par(mfrow =c(2,1))
# plot(q_acc_y, type = "l", col = "green", main = "collision")
# points(q_acc_x, type = "l", col = "red")
# points(q_acc_z, type = "l", col = "blue")
# 
# points(q2_acc_y, type = "l", lty = 2, col = "green", main = "collision")
# points(q2_acc_x, type = "l", lty = 2, col = "red")
# points(q2_acc_z, type = "l", lty = 2, col = "blue")
# 
# 
# plot(q_g_x, type = "l", col = "green", main = "collision")
# points(q_g_y, type = "l", col = "red")
# points(q_g_z, type = "l", col = "blue")
# 
# points(q2_g_x, type = "l", lty = 2, col = "green", main = "collision")
# points(q2_g_y, type = "l", lty = 2, col = "red")
# points(q2_g_z, type = "l", lty = 2, col = "blue")
# 
# dev.off()
#----------------------------------------------------------------------------------



# # obj = generateNNTrainingsDataWithFeatures2(data,windowSize = windowSize, numOfVars = 8, overlap = windowSize/2.0, c(-windowSize/2.0:windowSize/2.0))
# obj2 = generateNNTrainingsDataWithFeatures2(data,windowSize = windowSize, numOfVars = 8, overlap = 1, c(1:1))
# C = cbind(obj2$X,obj2$Y_noFrame)
# write.table(C, file = "NNtrainingMultiClassFeatures.csv", sep=",", col.names = T, row.names = F)
# 
# 
# 
# C2 = cbind(obj$X,obj2$X,obj$Y_noFrame)
# write.table(C2, file = "NNtrainingMultiClassRawAndFeatures.csv", sep=",", col.names = T, row.names = F)


# p = obj$X[which(obj$Y_noFrame == colClass),]
# q = colMeans(p, na.rm=TRUE)
# 
# p2 = obj$X[which(obj$Y_noFrame != colClass),]
# q2 = colMeans(p2, na.rm=TRUE)
# 
# pdf("weightsFeaures.pdf")
# par(mfrow =c(1,1))
# plot(q, main = "collision (b) vs no collision (r)", col = "blue")
# points(q2, col = "red")
# 
# dev.off()

if(ncol(data) == 12) {
  plotNN(data)
  
  
  # m_ = 0
  # mOb = 0
  # for(i in seq(0,1.0,0.01)){
  #   ob = NNWithThreshold(data,i, toleranceWindow = 100)
  # 
  #   # length(data$collision)
  #   # length(data$NNoutput)
  # 
  #   #ob = eval(data$collision, data$NNoutput, i, toleranceWindow = 100)
  # 
  #   if(m_ <= ob$maxCrit ){
  #     m_ = ob$maxCrit
  #     mOb = ob
  #   }
  # }
  
  mOb = NNWithThreshold(data,0.3, toleranceWindow = 100)
  
  conf = matrix(c(mOb$TN,mOb$FP,mOb$FN,mOb$TP), byrow = TRUE, nrow = 2, ncol = 2, dimnames = list(c("keine Kollision","Kollision") ,c("keine Kollision vorhergesagt", "Kollision vorhergesagt")))
  
  write.table(conf,file = paste("confusion_", mOb$th ,".csv", sep = ""), sep = ",", col.names=NA, quote=FALSE)

  #write.table(conf,file = "confusion.csv",  sep = ";", col.names = FALSE, row.names = FALSE)
  
}



#plotWithoutCollision(data, viewSize*10, 9500, c(-10,30), c(-90,20))



i = 5820
offSet = predictOffSet(data,i,viewSize = viewSize/2.0, 2)
changeCollisionInData(data,i,i+offSet, windowSize = windowSize, viewSize = viewSize, ind = ind, colClass = colClass, paste("TN_" ,i, sep = ""), FALSE)


