#config files for each DNN model


##FALLBACK

#About IsnodeSupported func in tensorflow_lite's delegation mechanism

![FALLBACK](https://github.com/easyhardhoon/FBF-TF-hoon/assets/89892717/55bd1556-01c1-4550-b45b-6299b347d456)

But there still exists indirect fallback in "above Supported layer", which does not expose "False" in check function.

Indiect fallback commonly exposes "error" at runtime, before interpreter invoke, But there are still cases where they do not.


#Hidden Fallback

For example, YOLOV4-tiny's ADD layer exposes error at runtime on jetson xavier-nx (using OPenGL backend). 

But on odroid xu4, which using OPenCL backend,does not expose error at runtime, but it casues a loss of accuracy. 

Also, lanenet's ADD layer (specific 63 node num) expoes error at runtime on nx, but on odroid, it does not expose error at runtime.

Maybe, this is a kind of bug in tensorflow lite's past version (2.4.1)

The only thing to do is add codes for applying custom fallback on IsnodeSupported func (FBF-TF/Work_Hoon) 
