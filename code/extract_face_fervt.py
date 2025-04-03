import torch
import numpy as np
import cv2
import os
import time
from model.FERVT import FERVT
import pickle
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from torch.cuda.amp import autocast
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = MTCNN(keep_all=False, device='cpu')
fer_vt = FERVT(device=device).to(device)
features = []

def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())
    del output

handle = fer_vt.vta.layernorm.register_forward_hook(hook_fn)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def save_features_as_p(video_features, video_name, folder_name):
    output_dir = f'/root/autodl-tmp/features/extracted_features_without_xml/{folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{video_name}.p')
    non_zero_features = [feature for feature in video_features if np.any(feature)]
    data = {"features": non_zero_features} if non_zero_features else {"all_zero": True}
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"特征已保存到 {output_path}")

def generate_sample_indices(total_frames):
    sample_count = max(1, total_frames // 10)
    sample_interval = total_frames / sample_count
    return [int(i * sample_interval) for i in range(sample_count)]

def process_frame_batch(frames, frame_ids):
    batch_features = []
    for i, frame in enumerate(frames):
        faces = detector(frame)
        if faces is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
            features.clear()
            try:
                with autocast():
                    with torch.no_grad():
                        _ = fer_vt(frame_tensor)
                extracted_feature = features[0] if features else np.zeros((1, 192))
                batch_features.append(extracted_feature)
                print(f"视频 {frame_ids[i]} 帧检测到人脸并提取特征成功，特征大小: {extracted_feature.shape}")
            except Exception as e:
                print(f"提取特征时出错: {e}")
                batch_features.append(np.zeros((1, 192)))
        else:
            batch_features.append(np.zeros((1, 192)))
    return batch_features

def process_video_batch(video_file, batch_size=4, folder_name=''):
    video_name = os.path.basename(video_file).split('.')[0]
    p_file_path = os.path.join(f'/root/autodl-tmp/features/extracted_features_without_xml/{folder_name}', f'{video_name}.p')
    if os.path.exists(p_file_path):
        print(f"已存在特征文件，跳过视频 {video_file}")
        return

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_features, frame_buffer, frame_ids = [], [], []
    sample_indices = generate_sample_indices(total_frames)
    print(f"正在批处理视频 {video_file}")

    for frame_id in tqdm(sample_indices, desc=f"处理 {video_name} 帧", total=len(sample_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)
            frame_ids.append(frame_id)
            if len(frame_buffer) == batch_size or frame_id == sample_indices[-1]:
                batch_features = process_frame_batch(frame_buffer, frame_ids)
                video_features.extend(batch_features)
                frame_buffer, frame_ids = [], []
        else:
            print(f"无法读取视频 {video_file} 的第 {frame_id} 帧")
            video_features.append(np.zeros((1, 192)))

    cap.release()
    print(f"完成处理视频 {video_file}")
    save_features_as_p(video_features, video_name, folder_name)

def process_all_videos(video_dir):
    folder_names = os.listdir(video_dir)
    for folder_name in tqdm(folder_names, desc="处理文件夹"):
        folder_path = os.path.join(video_dir, folder_name)
        if os.path.isdir(folder_path):
            video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            total_videos = len(video_files)
            for idx, video_file in enumerate(tqdm(video_files, desc=f"处理 {folder_name}", leave=False)):
                video_file_path = os.path.join(folder_path, video_file)
                if os.path.isfile(video_file_path):
                    print(f"处理 {folder_name} 的第 {idx + 1}/{total_videos} 个视频: {video_file}")
                    process_video_batch(video_file_path, folder_name=folder_name)
                else:
                    print(f"视频文件缺失或格式不正确: {video_file_path}")

if __name__ == "__main__":
    video_dir = '/root/autodl-tmp/PCLMM'
    process_all_videos(video_dir)
