#!/bin/bash

helpFunction()
{
   echo "script to initialize the workspace"
   echo "Usage: $0 [-d data_set_zip_file] [-m]"
   echo -e "\t-d to unzip the dataset in the dataset directory"
   echo -e "\t-m to dowload pretrained model"
   echo -e "\t-h to print the usage of this script"
   exit 1 # Exit script after printing help
}
dowloadModels=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    -d) dataSetZipFile="$2"; shift 2;;
    -m) dowloadModels=true; shift 1;;
	-h) helpFunction;;
    -*) echo "unknown option: $1" >&2;helpFunction;;
    *) handle_argument "$1"; shift 1;;
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
	echo "PUT THE DATA SET INSIDE dataset directory"
fi
if [ "$dowloadModels" == "true" ]
then
	echo "dowload pretrained model..."
	wget "https://drive.google.com/uc?export=download&id=1rluJbQVEFLRjxHqcM75M1WT29-dFw1Cq"
	mkdir -p ./Models/test9
	mv "uc?export=download&id=1rluJbQVEFLRjxHqcM75M1WT29-dFw1Cq" ./Models/test9/test9.h5
	echo "dowload finished"
	echo "to test the trained model lauch test.py --model_name test9"
fi