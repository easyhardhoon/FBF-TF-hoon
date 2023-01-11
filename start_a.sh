#!/bin/bash

UnitSimple="/home/nvidia/FBF-TF-hoon"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite Unit_simple Test"
echo $DISPLAY

. ${TflitePath}/build_aarch64_lib.sh
touch minimal.cc
make

