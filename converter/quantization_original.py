import tensorflow as tf
from tensorflow import keras
import numpy as np

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/11')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open("1_model.tflite", "wb").write(tflite_quant_model)
print("succes")
