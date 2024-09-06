import os
import pickle
import librosa
import numpy as np
from tqdm import tqdm

# Helper function to generate MFCCs
def extract_mfcc(path):
    audio, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 原始音频文件夹路径
AUDIO_FOLDER_ROOT = '/root/MFCC/AUDIO_wav'  # 替换为你的音频文件夹路径

# 特征保存文件夹路径
FEATURE_FOLDER_ROOT = '/root/autodl-tmp/AUDIO_features'  # 替换为你想要保存特征文件的文件夹路径
os.makedirs(FEATURE_FOLDER_ROOT, exist_ok=True)  # 如果文件夹不存在，则创建它

# 遍历每个子文件夹并处理其中的WAV文件
for subfolder in os.listdir(AUDIO_FOLDER_ROOT):
    subfolder_path = os.path.join(AUDIO_FOLDER_ROOT, subfolder)

    if os.path.isdir(subfolder_path):  # 确保这是一个目录
        # 创建相应的输出文件夹
        feature_subfolder = os.path.join(FEATURE_FOLDER_ROOT, subfolder)
        os.makedirs(feature_subfolder, exist_ok=True)

        # 获取子文件夹中的所有 WAV 文件
        wav_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]

        # 提取特征并保存为 .p 文件
        for wav_file in tqdm(wav_files):
            try:
                # 构建输出文件路径
                output_file = os.path.join(feature_subfolder, wav_file.replace('.wav', '.p'))

                # 检查是否已经处理过这个音频文件
                if os.path.exists(output_file):
                    print(f"特征已经存在，跳过: {output_file}")
                    continue

                # 提取音频特征
                mfcc_feature = extract_mfcc(os.path.join(subfolder_path, wav_file))

                # 保存特征为 .p 文件到指定的特征文件夹
                with open(output_file, 'wb') as fp:
                    pickle.dump(mfcc_feature, fp)

                print(f"成功提取并保存特征: {output_file}")

            except Exception as e:
                print(f"处理文件 {wav_file} 时出错: {e}")
