import json
import os

def split_json_file(file_path, output_folder, num_chunks):
    with open(file_path, 'r') as file:
        data = json.load(file)

        if 'images' in data and 'annotations' in data:
            images = data['images']
            annotations = data['annotations']

            total_records = len(images)
            chunk_size = total_records // num_chunks
            remainder = total_records % num_chunks

            start_idx = 0
            for i in range(num_chunks):
                chunk_records = chunk_size
                if i < remainder:
                    chunk_records += 1

                end_idx = start_idx + chunk_records
                chunk_images = images[start_idx:end_idx]
                chunk_annotations = [ann for ann in annotations if ann['image_id'] in [img['id'] for img in chunk_images]]
                start_idx = end_idx

                chunk_data = {
                    'info': data['info'],
                    'images': chunk_images,
                    'licenses': data['licenses'],
                    'annotations': chunk_annotations,
                    'categories': data['categories']
                }

                # 분할된 데이터를 새로운 JSON 파일로 저장
                output_file = f"{output_folder}/chunk_{i}.json"
                with open(output_file, 'w') as outfile:
                    json.dump(chunk_data, outfile)

        else:
            raise KeyError("No 'images' or 'annotations' key found in the JSON data.")

# JSON 파일 경로
file_path = '/home/nvidia/coco_raw/instances_val2014.json'

# 분할된 데이터를 저장할 폴더 경로
output_folder = '/home/nvidia/coco_raw/tmp'

# 분할할 개수
num_chunks = 10

# JSON 파일 분할
split_json_file(file_path, output_folder, num_chunks)

