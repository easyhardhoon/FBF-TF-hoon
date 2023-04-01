# tflite models

mnist_1.tflite -> original model 

mnist_2.tflite -> change hidden layer's activation [relu->leaky_relu]

mnist_3.tflite -> remove concatenate layer and add dropout layer at that point

mnist_4.tflite -> remove dropout layer & change leaky_relu to elu [activation_function]

mnist_5.tflite -> change first Fully-Connected Layer's activation func [leaky_relu -> relu]

mnist_6.tflite -> custom class model including add & mul layer

mnist_7.tflite -> custom class model including add & mul layer

mnist_8.tflite -> custom class model including SPLIT layer

mnist_9.tflite -> change hidden layer's activation [relu->leaky_relu] based mnist_8.tflite

mnist_10.tflite -> tuning model based on VGG16 

mnist_11.tftlie -> add dummy concate layer on mnist_10.tflite 

# main models

1. mnist_9.tflite -> originally Fallback model by custom class using SPLIT layer

2. mnist_11.tflite -> add additionally Fallback layer on tuning VGG16 model

3. yolo.tflite -> yolov4-tiny model. SPLIT layer is FALLBACK layer that forces to fall back to CPU

# main flow

1. minimal_cpu -> test minimal example on single unit CPU 

2. minimal_gpu -> test minimal example on single unit GPU

3. minimal_cpu_gpu -> test minimal example on multiple unit, CPU & GPU without any data exchange

4. unit_simple -> test minimal example on multiple unit, CPU & GPU with data exchange


# delegation node partitoning cases (***)

(activation function is generally within other major layer (ex CONV, Fully-Connected), but there is exceptional case for memory optimization)

1. mnist_4.tlite (case FALLBACK)

conv elu pool conv elu pool conv elu flatt fc elu fc softmax

  T   F   T     T   F    T    T   F    T   T   F  T     T    (case : activation)

  F   T   T     F   T    T    F   T    T   T   T  T     T    (case : CONV)

  T   T   F     T   T    F    T   T    T   T   T  T     T    (case : pool)

  T   T   T     T   T    T    T   T    T   F   T  F     T    (case : FC)  

2. mnist_5.tlite (case FALLBACK)

conv elu pool conv elu pool conv elu flatt fc fc softmax

  T   F   T     T   F    T    T   F    T   T   T     T    (case : activation)

  F   T   T     F   T    T    F   T    T   T   T     T    (case : CONV)

  T   T   F     T   T    F    T   T    T   T   T     T    (case : pool)

  T   T   T     T   T    T    T   T    T   F   F     T    (case : FC)  


# mnist_9.tflite (final test model) (by custom class)

split --> conv pool flatten FC

                                +---> concatenate FC

      --> conv pool flatten FC

this model is related to "Input data partitioning"

split layer is unsupported layer in opengl delegation, with tensorflow lite version 2.4.1


# mnist_11.tflite (add dummy concate layer that causes fallback  on tuning VGG16 model)

