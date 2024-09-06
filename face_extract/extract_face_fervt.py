import torch
import numpy as np
import cv2
import os
import time
from model.FERVT import FERVT  # 确保导入模型
import pickle
from torchvision import transforms
from PIL import Image
from mtcnn import MTCNN
from torch.cuda.amp import autocast  # 用于混合精度
from tqdm import tqdm  # 用于进度条

# 初始化 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 MTCNN 人脸检测器
detector = MTCNN()

# 初始化 FER-VT 模型，并将其加载到 GPU 上
fer_vt = FERVT(device=device).to(device)

# 用于存储提取的特征
features = []


# 钩子函数用于捕获所需层的输出
def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())


# 注册钩子到 VTA 的 LayerNorm 层，提取 Transformer 输出的特征
handle = fer_vt.vta.layernorm.register_forward_hook(hook_fn)

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def save_features_as_p(video_features, video_name, folder_name):
    """将提取的特征保存为 .p 文件"""
    output_dir = f'/root/autodl-tmp/extracted_features_without_xml/{folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{video_name}.p')

    non_zero_features = [feature for feature in video_features if np.any(feature)]

    if non_zero_features:
        data = {"features": non_zero_features}
    else:
        data = {"all_zero": True}

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"特征已保存到 {output_path}")


# 直接生成采样帧索引，不保存到文件
def generate_sample_indices(total_frames):
    """生成并返回采样帧的索引"""
    sample_count = max(1, total_frames // 10)  # 确保至少采样一帧
    sample_interval = total_frames / sample_count
    return [int(i * sample_interval) for i in range(sample_count)]


def process_frame_batch(frames, frame_ids):
    """
    处理一批帧，提高处理效率
    """
    batch_features = []
    for i, frame in enumerate(frames):
        faces = detector.detect_faces(frame)
        if faces:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

            features.clear()
            try:
                with autocast():  # 混合精度
                    with torch.no_grad():
                        _ = fer_vt(frame_tensor)
                extracted_feature = features[0] if features else np.zeros((1, 192))  # 更新了特征的维度大小
                batch_features.append(extracted_feature)
                print(f"视频 {frame_ids[i]} 帧检测到人脸并提取特征成功，特征大小: {extracted_feature.shape}")
            except Exception as e:
                print(f"提取特征时出错: {e}")
                batch_features.append(np.zeros((1, 192)))  # 若出错，也填充零向量
        else:
            batch_features.append(np.zeros((1, 192)))  # 若无特征，则填充零向量
    return batch_features


def process_video_batch(video_file, batch_size=16, folder_name=''):
    """
    以批处理方式处理单个视频，提高处理效率
    """

    video_name = os.path.basename(video_file).split('.')[0]
    # 检查是否已经存在对应的 .p 文件
    p_file_path = os.path.join(f'/root/autodl-tmp/extracted_features_without_xml/{folder_name}', f'{video_name}.p')
    if os.path.exists(p_file_path):
        print(f"已存在特征文件，跳过视频 {video_file}")
        return  # 跳过该视频

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_features = []
    frame_buffer = []
    frame_ids = []

    # 生成采样帧的索引
    sample_indices = generate_sample_indices(total_frames)
    print(f"正在批处理视频 {video_file}")

    # 添加处理帧时的进度条
    for frame_id in tqdm(sample_indices, desc=f"处理 {video_name} 帧", total=len(sample_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 跳转到指定帧
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)
            frame_ids.append(frame_id)

            if len(frame_buffer) == batch_size or frame_id == sample_indices[-1]:
                batch_features = process_frame_batch(frame_buffer, frame_ids)
                video_features.extend(batch_features)

                frame_buffer = []
                frame_ids = []
        else:
            print(f"无法读取视频 {video_file} 的第 {frame_id} 帧")
            video_features.append(np.zeros((1, 192)))  # 若读取失败，填充零向量

    cap.release()
    print(f"完成处理视频 {video_file}")

    save_features_as_p(video_features, video_name, folder_name)


def process_all_videos(video_dir):
    """
    处理所有视频，跳过已有的文件
    """
    folder_names = os.listdir(video_dir)

    for folder_name in tqdm(folder_names, desc="处理文件夹"):
        folder_path = os.path.join(video_dir, folder_name)
        if os.path.isdir(folder_path):
            # 获取当前文件夹下的视频文件
            video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            total_videos = len(video_files)
            for idx, video_file in enumerate(tqdm(video_files, desc=f"处理 {folder_name}", leave=False)):
                video_file_path = os.path.join(folder_path, video_file)
                if os.path.isfile(video_file_path):
                    print(f"处理 {folder_name} 的第 {idx + 1}/{total_videos} 个视频: {video_file}")
                    process_video_batch(video_file_path, folder_name=folder_name)
                else:
                    print(f"视频文件缺失或格式不正确: {video_file_path}")


# 主函数
if __name__ == "__main__":
    video_dir = '/root/autodl-tmp/video'  # 替换为存储视频文件的根目录
    process_all_videos(video_dir)
