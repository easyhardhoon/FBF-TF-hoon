import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

model = tf.keras.models.load_model("yolov4-tiny-416.h5")

log_dir = './logs'  
tensorboard_callback = TensorBoard(log_dir=log_dir)

model.compile(run_eagerly=False)

model.build(input_shape=(None, 416, 416, 3))

dummy_data = tf.random.normal((100, 416, 416, 3))
dummy_labels = tf.random.uniform((100, 7, 7, 30))

model.fit(dummy_data, dummy_labels, epochs=100, batch_size=32, callbacks=[tensorboard_callback])

with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.trace_on(graph=True, profiler=True)
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)

