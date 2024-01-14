#!/bin/bash

minimal_hoon="/home/nvidia/FBF-TF-hoon"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"

echo "@@@re-build gpu_delegate.so AND make for tensorflowlite_test@@@"

cd ../../FBF-TF 
sudo bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

cd ../FBF-TF-hoon
sudo ldconfig

export DISPLAY=:0
#xrandr
#glxinfo | grep "OpenGL version"

. ${TflitePath}/build_aarch64_lib.sh
touch ../FBF-TF/tensorflow/lite/unit.cc

cd APP_DOT
make nx


# if just get out tflitepath, go to ../FBF-TF/tensorflow/lite/tools/make and make clean 
