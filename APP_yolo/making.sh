#!/bin/bash

TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"

echo "@@@re-build gpu_delegate.so AND make for tensorflowlite_test@@@"

cd ../../FBF-TF 
sudo bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

cd ../FBF-TF-hoon
sudo ldconfig
export DISPLAY=:10

. ${TflitePath}/build_aarch64_lib.sh
cd APP_yolo
make

