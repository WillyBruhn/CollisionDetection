#!/usr/bin/env bash

targetFile="NNtrainingMultiClassFeatures.csv"
targetFile=$1

findings=$(find -iname "$targetFile")

echo $findings

arr=($findings)

out="NNtrainingMultiClassFeaturesALL.csv"
out=$2

head -n 1 ${arr[0]} > $out

suu=0
for i in $findings; do
	tail -n+2 $i >> $out
	val=$(wc -l $i)
	echo $val

	#suu=$((suu + val))
done

wc -l $out

#echo $suu
