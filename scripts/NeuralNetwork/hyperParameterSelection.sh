#!/bin/bash


epochs=(100)

learnRates=(0.005)
batchSizes=(2000) # 200
regularizations=(0.0000000001) # unweighted ganz gut

regularizations=(0.0000000)

trainSplit=0.001

iterations=1

modelChoice=1

#pa="/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/"
#pa="/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/TrainingMultiClass/"

pa="/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/SimpleThVSNN_final/Training/"


#dataSet="$pa/NNtrainingALLNoFrame.csv"

dataSet="$pa/NNtrainingALL.csv"


#dataSet="$pa/NNtrainingALLMultiClassNoFrame.csv"

confMatName="$pa/confusionNN_Train.csv"

for epoch in ${epochs[@]}; do
	for learnRate in ${learnRates[@]}; do
		for batchSize in ${batchSizes[@]}; do
			for regularization in ${regularizations[@]}; do
				call="python3 NN4.py --epochs $epoch --learnRate $learnRate --batchSize $batchSize --iterations $iterations --regularization $regularization --l1 2 --l2 2 --modelChoice 1 --dataSet $dataSet --TrainTestSplitRatio $trainSplit --confMat $confMatName --splitRandom True"

				
				b=3
				if [ $modelChoice -eq "$b" ]
				then
					call="python3 NN4.py --epochs $epoch --learnRate $learnRate --batchSize $batchSize --iterations $iterations --regularization $regularization --l1 5 --l2 50 --modelChoice 3 --dataSet $dataSet --TrainTestSplitRatio $trainSplit"
				fi

				b=4
				if [ $modelChoice -eq "$b" ]
				then
					call="python3 NN4.py --epochs $epoch --learnRate $learnRate --batchSize $batchSize --iterations $iterations --regularization $regularization --l1 10 --l2 10 --l3 10 --l4 10 --l5 10 --modelChoice 3 --dataSet $dataSet --TrainTestSplitRatio $trainSplit"
				fi

				$call
			done
		done;
	done;
done;

