import tensorflow as tf
from tensorflow import keras
import numpy as np



def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1,28,28,1)
        yield [data.astype(np.float32)]



converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/new_cpu')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

open("new_cpu", "wb").write(tflite_quant_model)
print("succes")
