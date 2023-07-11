import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('./checkpoints/yolov4-tiny-416')


anchors = np.array([[x, y] for x in [10, 14, 23, 27, 37, 58, 81, 82, 135, 169]
                    for y in [10, 14, 23, 27, 37, 58, 81, 82, 135, 169]])

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS,
                                       'TFLite_Detection_PostProcess']

# Enable experimental new converter and new quantizer
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True

# Set the anchor parameters as an input tensor
converter.allow_custom_ops = True
converter.input_tensors = {"input_anchors": anchors.astype(np.float32)}

tflite_model = converter.convert()
with open('./checkpoints/yolov4-tiny-customOP.tflite', 'wb') as f:
    f.write(tflite_model)

