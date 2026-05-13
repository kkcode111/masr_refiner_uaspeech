import os

def generate_annotation():
    """
    遍历新数据集文件夹并生成符合 MASR 要求的 .txt 标注文件。
    文件夹结构：dataset/audio/processed_fishaudio/{level}_audio_processed/
    文件名格式：{level}_{text}_{index}_processed.wav
    """
    audio_root = 'dataset/audio/processed_fishaudio'
    output_train = 'dataset/annotation/train_new.txt'
    output_test = 'dataset/annotation/test_new.txt'
    
    # 确保标注目录存在
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    
    train_count = 0
    test_count = 0
    print(f"正在扫描目录: {audio_root} ...")
    
    with open(output_train, 'w', encoding='utf-8') as f_train, \
         open(output_test, 'w', encoding='utf-8') as f_test:
        
        # 遍历 processed_fishaudio 下的所有子目录
        for root, dirs, files in os.walk(audio_root):
            for file in files:
                if file.endswith('.wav'):
                    # 文件名例子: 
                    # 训练集: high_a_3_processed.wav
                    # 测试集: high_astounded_B2_10_processed.wav
                    parts = file.split('_')
                    
                    if len(parts) >= 4:
                        text = parts[1]
                        # 获取相对于项目根目录的相对路径，并统一使用正斜杠
                        rel_path = os.path.join(root, file).replace('\\', '/')
                        
                        # 划分逻辑：参考 UASpeech 标准
                        # 如果文件名中包含 B2，则归类为测试集，否则归类为训练集
                        if '_B2_' in file:
                            f_test.write(f"{rel_path}\t{text}\n")
                            test_count += 1
                        else:
                            f_train.write(f"{rel_path}\t{text}\n")
                            train_count += 1
                    else:
                        print(f"跳过格式不正确的音频文件: {file}")
    
    print(f"处理完成！")
    print(f"新训练集 (非 B2): {train_count} 条 -> {output_train}")
    print(f"新测试集 (B2): {test_count} 条 -> {output_test}")

if __name__ == "__main__":
    generate_annotation()
