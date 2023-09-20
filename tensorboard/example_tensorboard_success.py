import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import datetime

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './checkpoints_summary/logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def create_model():
    network = keras.Sequential()
    network.add(keras.layers.Flatten(input_shape=(28, 28)))
    network.add(keras.layers.Dense(512, activation='relu', input_shape=(28*28,), name="layer1"))
    network.add(keras.layers.Dense(10, activation='softmax', name="layer2"))
    return network

@tf.function
def forward_pass(model, inputs):
    return model(inputs, training=False)

model = create_model()
model.summary()

tf.summary.trace_on(graph=True, profiler=True)
predictions = forward_pass(model, train_images[:2])
print(predictions)
with train_summary_writer.as_default():
    tf.summary.trace_export(name="network_trace", step=0, profiler_outdir=train_log_dir)

