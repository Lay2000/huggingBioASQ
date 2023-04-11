import os
import json


for subdir, _, files in os.walk("configs/"):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # 修改目标属性
            json_data["hyperparameters"]["lr"] = 4.54e-6

            # 将修改后的json数据写回文件
            new_file_path = os.path.join(subdir, file.split(".")[0] + "_lr_4.54e-6.json")
            with open(new_file_path, 'w') as f:
                json.dump(json_data, f, indent=4)
