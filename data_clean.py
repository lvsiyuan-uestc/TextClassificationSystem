import os
import jieba
import re
import time

# 路径设置
input_dir = './cnews'  # 数据文件夹路径
output_dir = './processed_data'  # 清洗和分词后的输出路径
os.makedirs(output_dir, exist_ok=True)


# 正则表达式用于清洗文本
def clean_text(text):
    # 去除标点符号和特殊字符，只保留中英文和数字
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)  # 去除多余空格
    return text.strip()


# 分词函数
def preprocess_and_segment_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue
            try:
                label, content = line.strip().split('\t', 1)  # 分割标签和内容
            except ValueError:
                continue  # 如果某行格式错误跳过

            content = clean_text(content)
            words = jieba.lcut(content)
            segmented_text = ' '.join(words)
            fout.write(f'{label}\t{segmented_text}\n')


#分别处理训练集、验证集、测试集
for filename in ['cnews.train.txt', 'cnews.val.txt', 'cnews.test.txt']:
    start_time=time.time()
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace('.txt', '_seg.txt'))
    preprocess_and_segment_file(input_path, output_path)
    end_time=time.time()
    print(f'处理完成：{filename} -> {output_path},耗时{end_time-start_time}s')
