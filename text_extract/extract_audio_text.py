import whisper
import os

def transcribe_multilingual_audio_and_save(audio_path, output_path):
    # 加载 Whisper 模型 (base, small, medium, large)
    model = whisper.load_model("large")

    # 转录音频文件
    result = model.transcribe(audio_path, task="transcribe")

    # 获取转录的文本
    transcribed_text = result["text"]

    # 将转录的文本保存到文件
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(transcribed_text)
    print(transcribed_text)

    print(f"识别结果已保存到 {output_path}")

def process_all_audios(input_folder_root, output_folder_root):
    # 遍历输入文件夹中的所有子文件夹和音频文件
    for subfolder in os.listdir(input_folder_root):
        subfolder_path = os.path.join(input_folder_root, subfolder)

        if os.path.isdir(subfolder_path):  # 确保这是一个目录
            output_folder = os.path.join(output_folder_root, subfolder)
            os.makedirs(output_folder, exist_ok=True)  # 确保输出子文件夹存在

            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.wav'):
                    audio_path = os.path.join(subfolder_path, file_name)
                    output_path = os.path.join(output_folder, file_name.replace('.wav', '.txt'))

                    # 检查是否已经处理过这个音频文件
                    if os.path.exists(output_path):
                        print(f"Transcription already exists for {file_name}, skipping...")
                        continue

                    # 转录并保存文本
                    transcribe_multilingual_audio_and_save(audio_path, output_path)

# 示例用法
input_folder_root = "/root/MFCC/AUDIO_wav"  # 输入音频文件夹路径
output_folder_root = "/root/MFCC/TEXT_txt"  # 输出文本文件夹的根路径

process_all_audios(input_folder_root, output_folder_root)
