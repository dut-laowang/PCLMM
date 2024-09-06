import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, classification_report
from sklearn.metrics import confusion_matrix, classification_report


# 自定义 collate_fn 以处理不等长序列
def custom_collate_fn(batch):
    audio_batch = [item[0].clone().detach() for item in batch]
    video_batch = [item[1].clone().detach() for item in batch]
    face_batch = [item[2].clone().detach() for item in batch]
    labels = torch.tensor([item[3] for item in batch])

    audio_batch = pad_sequence(audio_batch, batch_first=True)
    video_batch = pad_sequence(video_batch, batch_first=True)
    face_batch = pad_sequence(face_batch, batch_first=True)

    return audio_batch, video_batch, face_batch, labels

# 自定义数据集类，用于加载已经提取好的特征
class CustomDataset(Dataset):
    def __init__(self, audio_data, video_data, face_data, labels):
        self.audio_data = audio_data
        self.video_data = video_data
        self.face_data = face_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = torch.tensor(np.array(self.audio_data[idx])).float().cuda()
        video = torch.tensor(np.array(self.video_data[idx])).float().cuda()
        face = torch.tensor(np.array(self.face_data[idx])).float().cuda()
        label = torch.tensor(self.labels[idx]).float().cuda()
        return audio, video, face, label

# 递归遍历所有子文件夹
def find_all_files(root_folder, extension='.p'):
    files = []
    for root, dirs, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

# 处理从.p文件中加载的特征
def process_loaded_features(loaded_feature):
    if "all_zero" in loaded_feature:
        return np.zeros((1, 192))
    else:
        return np.array(loaded_feature["features"]).astype(np.float32)

# 加载特征和标签的函数
def load_features_and_labels(audio_folder, video_folder, face_folder, annotation_csv):
    try:
        annotations = pd.read_csv(annotation_csv)
    except Exception as e:
        raise RuntimeError(f"读取 {annotation_csv} 文件时出错: {e}")

    audio_features = []
    video_features = []
    face_features = []
    labels = []
    subsets = []
    missing_modalities = []

    # 获取所有模态的文件路径
    audio_files = find_all_files(audio_folder)
    video_files = find_all_files(video_folder)
    face_files = find_all_files(face_folder)

    # 遍历表情特征文件夹中的每个文件
    for face_path in face_files:
        base_name = os.path.basename(face_path).replace('.p', '')

        # 匹配文件名，确保与Annotation中的File列匹配
        row = annotations.loc[annotations['File'] == base_name]
        if (row.empty):
            print(f"文件 {base_name} 在 Annotation.csv 中未找到，跳过...")
            continue

        # 找到对应的其他模态的文件路径
        audio_path = next((f for f in audio_files if os.path.basename(f).replace('.p', '') == base_name), None)
        video_path = next((f for f in video_files if os.path.basename(f).replace('.p', '') == base_name), None)

        # 检查所有模态的文件是否都存在
        if audio_path and video_path and face_path:
            try:
                with open(audio_path, 'rb') as f:
                    audio_feature = pickle.load(f)
                with open(video_path, 'rb') as f:
                    video_feature = pickle.load(f)
                with open(face_path, 'rb') as f:
                    face_feature = pickle.load(f)

                face_feature = process_loaded_features(face_feature)

                label = row['Annotation'].values[0]
                subset = row['Subset'].values[0].strip().lower()  # 获取Subset列信息

                audio_features.append(np.array(audio_feature).astype(np.float32))
                video_features.append(np.array(video_feature).astype(np.float32))
                face_features.append(face_feature)
                labels.append(label)
                subsets.append(subset)

            except Exception as e:
                print(f"处理文件 {base_name} 时出错: {e}")
        else:
            # 记录缺少模态的文件名
            missing_modalities.append(base_name)
            print(f"缺少模态文件: {base_name}")

    return audio_features, video_features, face_features, labels, subsets, missing_modalities

# 设置文件夹路径
audio_folder = '/root/autodl-tmp/AUDIO_features'
video_folder = '/root/autodl-tmp/VIT_features'
face_folder = '/root/autodl-tmp/extracted_features_without_xml'
annotation_csv = '/root/Annotation.csv'

# 加载特征和标签
audio_features, video_features, face_features, labels, subsets, missing_modalities = load_features_and_labels(
    audio_folder, video_folder, face_folder, annotation_csv)

# 打印缺少模态的文件名
if missing_modalities:
    print("缺少模态的文件名如下：")
    for filename in missing_modalities:
        print(filename)


# 检查加载的特征和标签数量
if len(audio_features) == 0 or len(video_features) == 0 or len(face_features) == 0:
    raise ValueError("加载的特征数量为0，请检查文件路径和文件名是否匹配")

print(f"加载的音频特征数量: {len(audio_features)}")
print(f"加载的视频特征数量: {len(video_features)}")
print(f"加载的表情特征数量: {len(face_features)}")
print(f"加载的标签数量: {len(labels)}")

# 根据 Subset 列划分训练集和测试集
train_indices = [i for i, s in enumerate(subsets) if s == 'train']
test_indices = [i for i, s in enumerate(subsets) if s == 'test']

# 获取训练集和测试集的数据
train_audio = [audio_features[i] for i in train_indices]
train_video = [video_features[i] for i in train_indices]
train_face = [face_features[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]

test_audio = [audio_features[i] for i in test_indices]
test_video = [video_features[i] for i in test_indices]
test_face = [face_features[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

# 构建训练集和验证集的数据集和数据加载器
train_dataset = CustomDataset(train_audio, train_video, train_face, train_labels)
test_dataset = CustomDataset(test_audio, test_video, test_face, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# 设置模型参数
embed_dim = 768  # 所有模态的目标嵌入维度
num_heads = 8
num_layers = 3

# 定义 Cross-Attention 模型
class MultiModalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, audio_dim=40, face_dim=192):
        super(MultiModalCrossAttention, self).__init__()

        self.audio_fc = nn.Linear(audio_dim, embed_dim)
        self.face_fc = nn.Linear(face_dim * 4, embed_dim)  # 将 4 个通道展平后输入

        self.audio_transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.video_transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.face_transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        self.cross_attention_audio_video = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention_audio_face = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention_video_face = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层，防止过拟合
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, audio_out, video_out, face_out):
        # 变换音频和表情特征维度
        audio_out = self.audio_fc(audio_out)

        # 将 Face_out 的 (channels, height, width) 展平成一维
        batch_size, seq_len, _, _, _ = face_out.size()
        face_out = face_out.view(batch_size, seq_len, -1)
        face_out = self.face_fc(face_out.float())

        # 确保所有输出的序列长度至少为1，必要时进行维度扩展
        if audio_out.dim() == 2:
            audio_out = audio_out.unsqueeze(1)
        if video_out.dim() == 2:
            video_out = video_out.unsqueeze(1)
        if face_out.dim() == 2:
            face_out = face_out.unsqueeze(1)

        # 平均池化所有输出，使其序列长度相同（例如 512）
        target_length = 512
        audio_out = F.adaptive_avg_pool1d(audio_out.transpose(1, 2), target_length).transpose(1, 2)
        video_out = F.adaptive_avg_pool1d(video_out.transpose(1, 2), target_length).transpose(1, 2)
        face_out = F.adaptive_avg_pool1d(face_out.transpose(1, 2), target_length).transpose(1, 2)

        for _ in range(self.num_layers):
            audio_out = self.audio_transformer(audio_out)
            video_out = self.video_transformer(video_out)
            face_out = self.face_transformer(face_out)

            audio_video_attn, _ = self.cross_attention_audio_video(audio_out, video_out, video_out)
            audio_face_attn, _ = self.cross_attention_audio_face(audio_out, face_out, face_out)
            video_face_attn, _ = self.cross_attention_video_face(video_out, face_out, face_out)

            audio_out = audio_out + audio_video_attn + audio_face_attn
            video_out = video_out + audio_video_attn + video_face_attn
            face_out = face_out + audio_face_attn + video_face_attn

            # 添加 Dropout
            audio_out = self.dropout(audio_out)
            video_out = self.dropout(video_out)
            face_out = self.dropout(face_out)

        combined_out = audio_out + video_out + face_out
        combined_out = torch.mean(combined_out, dim=1)
        output = self.fc(combined_out)

        return output


# 实例化模型
model = MultiModalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).cuda()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.BCEWithLogitsLoss()


# 评估指标函数
def eval_metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1_score = f1_score(y_true, y_pred, average='macro')
    auc_score = roc_auc_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "mf1_score": mf1_score,
        "auc": auc_score,
        "recall": recall,
        "precision": precision
    }


# 训练函数
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    all_labels = []
    all_preds = []

    for batch_idx, (audio, video, face, label) in enumerate(train_loader):
        audio, video, face, label = audio.to(device), video.to(device), face.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(audio, video, face)
        loss = criterion(output, label.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = torch.sigmoid(output).round().detach().cpu().numpy()
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds)

        if batch_idx % log_interval == 0:
            metrics = eval_metric(all_labels, all_preds)
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(audio)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}\tAccuracy: {metrics["accuracy"]:.4f}')


# 验证函数
def validation(model, device, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0

    with torch.no_grad():
        for audio, video, face, label in test_loader:
            audio, video, face, label = audio.to(device), video.to(device), face.to(device), label.to(device)
            output = model(audio, video, face)
            loss = criterion(output, label.float().unsqueeze(1))
            total_loss += loss.item()

            preds = torch.sigmoid(output).round().detach().cpu().numpy()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds)

    # 计算平均 loss
    average_loss = total_loss / len(test_loader)

    # 打印分类报告和混淆矩阵
    report = classification_report(all_labels, all_preds, target_names=["class 0", "class 1"], digits=4)
    print("Test F1 Report:\n", report)

    confusion = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", confusion)

    print(f'Validation set: Average loss: {average_loss:.4f}')

    return average_loss


# 训练循环
for epoch in range(20):  # 假设训练20个epoch
    train(20, model, device, train_dataloader, optimizer, epoch)
    validation(model, device, test_dataloader)
    scheduler.step()  # 调整学习率

# 保存模型
torch.save(model.state_dict(), 'multi_modal_cross_attention_model.pth')
