nx : yolo_with_DOT.cc
	g++ -o yolo_with_DOT yolo_with_DOT.cc -I/usr/include/opencv4\
		-pthread -g -I/home/nvidia/FBF-TF\
		-I/home/nvidia/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/include\
		-L/home/nvidia/FBF-TF/tensorflow/lite/tools/make/gen/linux_aarch64/lib\
		-I/home/nvidia/FBF-TF/tensorflow/lite/tools/make/downloads/absl\
		-L/home/nvidia/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/build\
		-lopencv_gapi\
		-I/home/nvidia/FBF-TF-hoon/APP_DOT\
		-ltensorflow-lite\
		-lflatbuffers /lib/aarch64-linux-gnu/libdl.so.2\
		-lopencv_stitching\
		-lopencv_highgui\
		-lopencv_dnn\
		-lopencv_videoio\
		-lopencv_ml -lopencv_video\
		-lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann\
		-lopencv_photo -lopencv_imgproc\
		-lopencv_core\
		/home/nvidia/FBF-TF/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so\
	 	/usr/lib/aarch64-linux-gnu/libEGL.so\
	        /usr/lib/aarch64-linux-gnu/libGL.so\
		/usr/lib/aarch64-linux-gnu/libGLESv2.so\

od : yolo_with_DOT.cc
	g++ -g -o yolo_with_DOT yolo_with_DOT.cc -I/usr/include/opencv4\
		-pthread -g -I/home/odroid/FBF-TF\
		-I/home/odroid/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/include\
		-L/home/odroid/FBF-TF/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
		-I/home/odroid/FBF-TF/tensorflow/lite/tools/make/downloads/absl\
		-L/home/odroid/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/build\
		-lopencv_gapi\
		-ltensorflow-lite\
		-I/home/odroid/FBF-TF-hoon/APP_DOT\
		-lflatbuffers /lib/arm-linux-gnueabihf/libdl.so.2\
		-lopencv_stitching\
		-lopencv_dnn\
		-lopencv_videoio\
		-lopencv_ml -lopencv_video\
		-lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann\
		-lopencv_photo -lopencv_imgproc\
		-lopencv_core\
		/home/odroid/FBF-TF/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so\
	 	/usr/lib/arm-linux-gnueabihf/libEGL.so\
	        /usr/lib/arm-linux-gnueabihf/libGL.so\
		/usr/lib/arm-linux-gnueabihf/libGLESv2.so\
                
