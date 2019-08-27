#!/bin/bash

helpFunction()
{
   echo "script to initialize the workspace"
   echo "Usage: $0 [-d data_set_zip_file]"
   echo -e "\t-d to unzip the dataset in the dataset directory"
   exit 1 # Exit script after printing help
}

while getopts "d:" opt
do
   case "$opt" in
      d ) dataSetZipFile="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

dataSetDir=./dataset
spectrogramDirV1=./dataset/spec/ver1/
spectrogramDirV2=./dataset/spec/ver2/
dataAugmentDir=./dataset/aug/
specV1AugmentDir=./dataset/aug/spec/ver1/
specV2AugmentDir=./dataset/aug/spec/ver2/
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