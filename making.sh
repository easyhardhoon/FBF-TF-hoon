#!/bin/bash

UnitSimple="/home/nvidia/FBF-TF-hoon"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite Test"
export DISPLAY=:0
xrandr
glxinfo | grep "OpenGL version"

. ${TflitePath}/build_aarch64_lib.sh
make

