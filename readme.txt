#tflite models
mnist_1.tflite -> original model 
mnist_2.tflite -> change hidden layer's activation [relu->leaky_relu]
mnist_3.tflite -> remove concatenate layer and add dropout layer at that point
mnist_4.tflite -> remove dropout layer & change leaky_relu to elu [activation_function]

#main flow
1. minimal_cpu -> test minimal example on single unit CPU 
2. minimal_gpu -> test minimal example on single unit GPU
3. minimal_cpu_gpu -> test minimal example on multiple unit, CPU & GPU without any data exchange
4. unit_simple -> test minimal example on multiple unit, CPU & GPU with data exchange

#unit_simple(***)
1.
2.
3.
4.

