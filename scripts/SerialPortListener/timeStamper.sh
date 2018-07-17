#!/bin/bash

echo "Enter a directory for the trainingsdata you want to generate:"
read dirName2
dirName="../../data/Training/$dirName2"
mkdir "Training"
mkdir $dirName

cd $dirName

fileName="collisions.txt"
rm $fileName

# calls the serialPort listener and retrieves 
# the data from the arduino
python3 listener.py &

echo "started serialport listener ..."

devName="/dev/ttyACM0"
stty -F $devName 2000000

keyPressed="k"
echo "Press 'enter' to register collision. Press 'q' and then 'enter' to exit."
while [ "$keyPressed" != "q" ]
do
	#read -p "Press enter to register collision" $keyPressed
	read keyPressed

	time=$(date +"%T:%3N")

	echo " --> collision at $time"
	echo $time >> $fileName

	#if [ "$keypressed" == "l"] then
	#	echo l > $devName
	#elif
	#	echo k > $devName
	#fi

	echo k > $devName
done

echo l > $devName

pkill -P $$

windowSize=20
frameSize=0

numOfVars=5

str="/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/Training/$dirName2"

#/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/./evluate.R $windowSize $frameSize $str $numOfVars

#xdg-open plots
#xdg-open TrainingsData
