import tensorflow as tf

# TensorFlow Lite 모델 파일 경로
model_path = '/path/to/your_model.tflite'

# TensorFlow Lite Interpreter를 생성하여 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 모델의 메타데이터 가져오기
metadata = interpreter.get_tensor_details()

# 라벨 맵 추출
label_map_tensor_index = 0  # 라벨 맵 텐서의 인덱스 (일반적으로 0 또는 다른 인덱스)
label_map = interpreter.get_tensor(metadata[label_map_tensor_index]['index'])

# 라벨 맵 파일로 저장
output_file = '/path/to/label_map.txt'
with open(output_file, 'w') as file:
    for label in label_map:
        file.write(label + '\n')

print(f"Label map saved to {output_file}")

