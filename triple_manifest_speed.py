import json
import random
import os

def generate_augmented_manifest(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    aug_lines = []
    print(f"开始处理 {len(lines)} 条原始数据...")
    
    for line in lines:
        data = json.loads(line)
        orig_duration = data['duration']
        
        # 1. 原始版本 (1.0x)
        data_orig = data.copy()
        data_orig['speed_rate'] = 1.0
        aug_lines.append(json.dumps(data_orig, ensure_ascii=False))
        
        # 2. 慢速版本 (0.9 ~ 0.95)
        data_slow = data.copy()
        slow_rate = round(random.uniform(0.9, 0.95), 3)
        data_slow['speed_rate'] = slow_rate
        # 语速变慢，时长增加
        data_slow['duration'] = round(orig_duration / slow_rate, 5)
        aug_lines.append(json.dumps(data_slow, ensure_ascii=False))
        
        # 3. 快速版本 (1.05 ~ 1.1)
        data_fast = data.copy()
        fast_rate = round(random.uniform(1.05, 1.1), 3)
        data_fast['speed_rate'] = fast_rate
        # 语速变快，时长缩短
        data_fast['duration'] = round(orig_duration / fast_rate, 5)
        aug_lines.append(json.dumps(data_fast, ensure_ascii=False))
        
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for l in aug_lines:
            f_out.write(l + '\n')
            
    print(f"处理完成！")
    print(f"总数据量: {len(aug_lines)} 条 (原数据的 3 倍)")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    generate_augmented_manifest('dataset/train.jsonl', 'dataset/train_aug.jsonl')
