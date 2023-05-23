import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

def get_flops(interpreter):
    flops = interpreter.get_tensor_details()[0]['quantization_parameters']['zero_points'][0]
    return flops

interpreter = Interpreter(model_path="yolov4-tiny-416.tflite")
interpreter.allocate_tensors()
flops = get_flops(interpreter)
printf("..", flops)
