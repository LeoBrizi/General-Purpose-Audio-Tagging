#!/bin/bash

helpFunction()
{
   echo "script to initialize the workspace"
   echo "Usage: $0 [-d data_set_zip_file]"
   echo -e "\t-d to unzip the dataset in the dataset directory"
   echo -e "\t-m to dowload pretrained models"
   exit 1 # Exit script after printing help
}

while getopts "d:m:" opt
do
   case "$opt" in
      d ) dataSetZipFile="$OPTARG" ;;
	  m ) dowloadModels=true ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

modelDir=./Models/
dataSetDir=./dataset/
spectrogramDirV1=./dataset/spec/ver1/
spectrogramDirV2=./dataset/spec/ver2/
dataAugmentDir=./dataset/aug/
specV1AugmentDir=./dataset/aug/spec/ver1/
specV2AugmentDir=./dataset/aug/spec/ver2/
echo "creating directory for network models"
mkdir -p $modelDir
echo "creating directory for data set"
mkdir -p $dataSetDir
echo "creating directories for spectrograms"
mkdir -p $spectrogramDirV1 $spectrogramDirV2
echo "creating directory for augmented data"
mkdir -p $dataAugmentDir $specV1AugmentDir $specV2AugmentDir
if [ "$dataSetZipFile" != "" ]
then
	echo "unzip the dataset inside dataset directory..."
	unzip $dataSetZipFile -d $dataSetDir
	trainZipFile = "audio_test.zip"
	testZipFile = "audio_train.zip"
	echo "unzip test files..."
	unzip "$dataSetDir/$testZipFile" -d $dataSetDir
	echo "unzip train files..."
	unzip "$dataSetDir/$trainZipFile" -d $dataSetDir
	echo "unziping finished."
else
	echo "put the data set inside $dataSetDir"
fi

if[ $dowloadModels ]
then
	echo "dowload pretrained models..."
	wget "https://drive.google.com/uc?export=download&id=1rluJbQVEFLRjxHqcM75M1WT29-dFw1Cq"
	mkdir -p ./Models/test9
	mv test9.h5 ./Models/test9/
	echo "dowload finished"
	echo "to test the trained model lauch test.py --model_name test9"
fi