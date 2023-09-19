import tensorflow as tf

saved_model = tf.keras.models.load_model("./yolov4-tiny-416")
saved_model.save("yolov4-tiny-416.h5")

