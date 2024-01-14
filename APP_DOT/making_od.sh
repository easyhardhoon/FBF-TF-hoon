#!/bin/bash

minimal_hoon="/home/odroid/FBF-TF-hoon"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/odroid/FBF-TF"

echo "@@@re-build gpu_delegate.so AND make for tensorflowlite_test@@@"

cd ../../FBF-TF 
bazel build -c opt tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so --copt -DEGL_NO_X11=1

cd ../FBF-TF-hoon
sudo ldconfig
. ${TflitePath}/build_bbb_lib.sh
cd ../FBF-TF-hoon/APP_DOT
make od


# if just get out tflitepath, go to ../FBF-TF/tensorflow/lite/tools/make and make clean 
