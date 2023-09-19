import tensorflow as tf

model = tf.keras.models.load_model("yolov4-tiny-416.h5")

input_data = tf.ones((1, 416, 416, 3))

tf.profiler.experimental.start('./logs')

predictions = model(input_data)

tf.profiler.experimental.stop()

