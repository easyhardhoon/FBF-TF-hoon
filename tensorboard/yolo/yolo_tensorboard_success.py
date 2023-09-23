import tensorflow as tf

saved_model_dir = './yolov4-tiny-416'
test_image = tf.ones((1, 416, 416, 3))
yolov4_tiny_model = tf.saved_model.load(saved_model_dir)

@tf.function
def forward_pass(model, inputs):
    return model(inputs, training=False)

model = yolov4_tiny_model
log_dir = './logs/yolov4-tiny'

tf.summary.trace_on(graph=True, profiler=True)
predictions = forward_pass(model,test_image)

with tf.summary.create_file_writer(log_dir).as_default():
    #yolov4_tiny_model(test_image)
    tf.summary.trace_export(name="yolov4-tiny_trace", step=0, profiler_outdir=log_dir)

