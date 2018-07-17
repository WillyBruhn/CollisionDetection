#!/bin/bash

#windowSizes=(10 15 20 25 30 35 40 45 50)
#frameSizes=(0 1 10 20 30 50 80 90 100 110 120 130 140 150 160 170 180 190 200)

windowSizes=(20)
frameSizes=(0)

pa=pwd


numOfVars=5		# with motor 5, without 3

for windowSize in ${windowSizes[@]}; do
    for frameSize in ${frameSizes[@]}; do

	echo $windowSize $frameSize

		#RCall="/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/./markCollisionsWithHand.R $windowSize $frameSize"
		RCall="$pa./markCollisionsWithHand.R"



		trainingsDir="../../data/Training"


		cd $trainingsDir
		directories=$(ls -d */)


		old=$(pwd)

		echo $directories

		count=1
		maxCount=$(wc -w <<< "$directories")

		for i in $directories; do
		{

			cd $i
			

			wd=$(pwd)
			#$RCall $wd $numOfVars &> "R.out"

			#$RCall $wd $numOfVars
			$RCall $wd $windowSize

			echo "$count / $maxCount : $i"
			
			count=$((count+1))

			cd $old
		} & 
		done

		wait

		

		plots=$(find -iname 'NNTrainingsDataPlot.*')

		#convert $plots plot.pdf
		#convert -dispose 2 -loop 0 -delay 25 $plots out.gif


		#composite  $plots overlay.pdf

		cd $old
		#mergerCall="../../joinExample/./mergeAllNNData.sh NNtrainingALL"$windowSize"_"$frameSize".csv"
		#$mergerCall

		#mergerCall2="../../joinExample/./mergeAllNNData.sh NNtrainingALL.csv"
		#$mergerCall2

		#../../joinExample/./mergeAllNNDataMultiClass.sh NNtrainingALLMultiClass.csv

		#../../joinExample/./mergeAllNNData.sh NNtrainingMultiClassRawAndFeatures.csv NNtrainingMultiClassRawAndFeaturesALL.csv

		#../../joinExample/./mergeAllNNData.sh NNtrainingMultiClassFeatures.csv NNtrainingMultiClassFeaturesALL.csv

		#../../joinExample/./mergeAllNNData.sh NNtrainingBlurredClass.csv NNtrainingBlurredClassALL.csv

		../../joinExample/./mergeAllNNData.sh NNtraining.csv NNtrainingALL.csv

		#../../joinExample/./mergeAllNNData.sh NNtraining.csv NNtrainingALL.csv

		cd "/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/"


		fname=$windowSize"_"$frameSize"_model.txt"
		#$res > $fname

		#NNcall="python3 ./NeuralNetwork/NN4.py"
		#$NNcall > $fname

		cd NeuralNetwork/

		#./hyperParameterSelection.sh

	done
done

exit
trainings="trainingsComparisons.txt"
rm $trainings
for i in $(find -iname "*model.txt"); do
	
	found=$(grep FN: $i)
	echo $i $found >> $trainings
done 

sort -k9 -n $trainings > "trainingsComparisonsSorted.txt"





