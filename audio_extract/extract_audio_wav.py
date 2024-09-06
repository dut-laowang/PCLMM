import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

def extract_audio_from_video(video_path, output_audio_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-ar', '44100',
        '-ac', '2',
        '-b:a', '192k',
        output_audio_path
    ]

    with open(os.devnull, 'w') as devnull:
        result = subprocess.run(command, stdout=devnull, stderr=devnull)

    if result.returncode != 0:
        print(f"Error extracting audio from {video_path}")
    else:
        print(f"Audio successfully extracted to: {output_audio_path}")

def process_video(video_info):
    video_path, output_audio_path = video_info
    if os.path.exists(output_audio_path):
        print(f"Audio already extracted for {os.path.basename(video_path)}, skipping...")
    else:
        extract_audio_from_video(video_path, output_audio_path)

def process_all_videos(input_folder, output_folder_root):
    video_files = []
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            output_folder = os.path.join(output_folder_root, subfolder)
            os.makedirs(output_folder, exist_ok=True)
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.mp4'):
                    video_path = os.path.join(subfolder_path, file_name)
                    output_audio_path = os.path.join(output_folder, file_name.replace('.mp4', '.wav'))
                    video_files.append((video_path, output_audio_path))

    with ThreadPoolExecutor(max_workers=4) as executor:  # 调整max_workers的值
        executor.map(process_video, video_files)

input_folder = "/root/autodl-tmp/video/"
output_folder_root = "/root/MFCC/"

process_all_videos(input_folder, output_folder_root)
