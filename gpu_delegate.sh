
cd ../FBF-TF 
sudo bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

cd ../FBF-TF-hoon
sudo ldconfig
