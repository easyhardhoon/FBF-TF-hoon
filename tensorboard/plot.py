import tensorflow as tf
from tensorflow.keras.utils import plot_model

# 모델 로드
model = tf.keras.models.load_model("yolov4-tiny-416.h5")

# 모델 구조 시각화 및 이미지로 저장
plot_model(model, to_file='model_structure.png', show_shapes=True)

