import os
import pickle
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


# Helper function to tokenize and extract features using BERT
def extract_text_features(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512,
                       add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0][0].detach().numpy()


# 原始文本文件夹路径
TEXT_FOLDER_ROOT = '/root/MFCC/TEXT_txt'  # 替换为你的txt文件所在的文件夹路径
# 特征保存文件夹路径
FEATURE_FOLDER_ROOT = '/root/autodl-tmp/TEXT_features'  # 替换为你想要保存特征文件的文件夹路径
os.makedirs(FEATURE_FOLDER_ROOT, exist_ok=True)  # 如果文件夹不存在，则创建它

# 加载预训练的 BERT 中文模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained("/root/MFCC/bert_chinese")
model = BertModel.from_pretrained("/root/MFCC/bert_chinese")

# 遍历每个子文件夹并处理其中的TXT文件
for subfolder in os.listdir(TEXT_FOLDER_ROOT):
    subfolder_path = os.path.join(TEXT_FOLDER_ROOT, subfolder)

    if os.path.isdir(subfolder_path):  # 确保这是一个目录
        # 创建相应的输出文件夹
        feature_subfolder = os.path.join(FEATURE_FOLDER_ROOT, subfolder)
        os.makedirs(feature_subfolder, exist_ok=True)

        # 获取子文件夹中的所有 TXT 文件
        txt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]

        # 提取特征并保存为 .p 文件
        for txt_file in tqdm(txt_files):
            try:
                # 构建输出文件路径
                output_file = os.path.join(feature_subfolder, txt_file.replace('.txt', '.p'))

                # 检查是否已经处理过这个文本文件
                if os.path.exists(output_file):
                    print(f"特征已经存在，跳过: {output_file}")
                    continue

                # 读取文本内容
                with open(os.path.join(subfolder_path, txt_file), 'r', encoding='utf-8') as file:
                    text = file.read()

                # 提取文本特征
                text_feature = extract_text_features(text, tokenizer, model)

                # 保存特征为 .p 文件到指定的特征文件夹
                with open(output_file, 'wb') as fp:
                    pickle.dump(text_feature, fp)

                print(f"成功提取并保存特征: {output_file}")

            except Exception as e:
                print(f"处理文件 {txt_file} 时出错: {e}")

print("所有文本特征提取完成并保存。")
