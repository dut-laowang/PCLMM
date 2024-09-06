import torch
import cv2
import os
import pickle
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
import time

# Initialize GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ViT model and load it to GPU
feature_extractor = ViTFeatureExtractor.from_pretrained("/root/VIT/googlevit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("/root/VIT/googlevit-base-patch16-224-in21k").to(device)

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def save_features_as_p(features, video_name, subfolder):
    """Save extracted features as .p file"""
    output_dir = f'/root/autodl-tmp/VIT_features/{subfolder}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{video_name}.p')
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {output_path}")

def process_video(video_file, subfolder):
    """Process a single video, extract features, and save them"""
    print(f"Processing video {video_file}")
    time.sleep(1)  # Pause for 1 second

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_features = []

    for frame_id in range(total_frames):
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)  # Preprocess image

            # Extract features
            with torch.no_grad():
                inputs = feature_extractor(images=frame_pil, return_tensors="pt").to(device)
                outputs = vit_model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]

                # Use average pooling to get the global feature representation
                extracted_feature = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                video_features.append(extracted_feature)
                print(f"Frame {frame_id} features extracted successfully, feature size: {extracted_feature.shape}")

        else:
            print(f"Cannot read frame {frame_id} of video {video_file}")

    cap.release()
    print(f"Completed processing video {video_file}")

    # Save current video's features to .p file
    save_features_as_p(video_features, os.path.basename(video_file).split('.')[0], subfolder)

def process_all_videos(video_root_dir):
    """Process all videos in the directory"""
    for subfolder in os.listdir(video_root_dir):
        subfolder_path = os.path.join(video_root_dir, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for video_file in os.listdir(subfolder_path):
                video_name = os.path.basename(video_file).split('.')[0]
                output_path = f'/root/autodl-tmp/VIT_features/{subfolder}/{video_name}.p'

                if os.path.isfile(output_path):  # Check if the output file already exists
                    print(f"Skipped already processed file: {video_file}")
                    continue

                video_path = os.path.join(subfolder_path, video_file)
                if os.path.isfile(video_path) and video_file.endswith('.mp4'):  # Assuming video files are in .mp4 format
                    process_video(video_path, subfolder)
                else:
                    print(f"Skipped non-video file: {video_file}")

# Main function
if __name__ == "__main__":
    video_root_dir = '/root/autodl-tmp/video'  # Root directory containing subfolders with video files
    process_all_videos(video_root_dir)
