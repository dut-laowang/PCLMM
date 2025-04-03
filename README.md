# PCLMM
 The code implementation for the article "Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Framework". The paper is **accepted by ICASSP 2025**, and the code is now released.

 The link for this paper is [https://ieeexplore.ieee.org/abstract/document/10890580](https://ieeexplore.ieee.org/abstract/document/10890580)
 
 a) Data Collection. Refining annotation guidelines and gathering data from Bilibili. 
 
 b) PCLMM dataset. A high-quality annotated dataset with PCL frame spans. 
 
 c) MultiPCL detector. A cross-attention mechanism.

 ![Our framework for this paper.](https://github.com/dut-laowang/PCLMM/blob/main/figure/P8.PNG)

# Updation
**February 9, 2025** – 🔥 The [Annotation_Link.csv](https://github.com/dut-laowang/PCLMM/blob/main/data/Annotation_Link.csv) containing the original video links has been updated.

**April 3, 2025**– 🔥 We have released the complete process for calling the PCLMM code.

# Dataset

The PCLMM dataset can be downloaded at [https://zenodo.org/records/15128981](https://zenodo.org/records/15128981)

The PCLMM dataset is sourced from Bilibili, the largest online community for young people in China. Our work aims to uncover microaggressions targeted at vulnerable groups, including discriminatory and patronizing language expressions (715 annotated videos). 

You can download `Annotation_Link.csv` for detailed annotation.
You can use `Annotation_Subset.csv` `Annotation.csv` for the experiment. 

> **Note:**  
> _The collection of this dataset follows Bilibili's Developer Agreement and Privacy Policy, with all data sourced from publicly available Bilibili video links. Please comply with Bilibili's relevant usage regulations when using this dataset to avoid any negative impact on the platform's services._


# Code

## Dataset
Download the PCLMM and place it at `/root/autodl-tmp/PCLMM`（**you can remove `/autodl-tmp` in your study**）.

## Video Feature Extraction

1. Download the model weights [https://huggingface.co/google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) and place them at `/root/autodl-tmp/code/googlevit-base-patch16-224-in21k`.
2. Run the following command to extract video features:
   
   ```bash
   python extract_video_vit.py
   ```

3. The extracted video features are saved by category in `/root/autodl-tmp/features/VIT_features`.

## Audio Feature Extraction

1. Install ffmpeg by running:

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

2. Run the following command to extract audio features:

   ```bash
   python extract_audio_wav.py
   ```

3. The extracted audio WAV files are saved in `/root/autodl-tmp/temp/WAV`.

4. To convert the WAV files into `.p` files, run:

   ```bash
   python MFCC.py
   ```

   The `.p` files are saved in `/root/autodl-tmp/features/AUDIO_features`.

## Text Feature Extraction

1. Download the BERT-Chinese model weights [https://huggingface.co/google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) and place them at `/root/autodl-tmp/code/bert_chinese`.
2. Run the following command to extract the transcribed text:

   ```bash
   python extract_audio_text.py
   ```

   The transcribed text is saved in `/root/autodl-tmp/temp/TXT`.

3. To extract text features, run:

   ```bash
   python BERT.py
   ```

   The extracted text features are saved in `/root/autodl-tmp/features/TEXT_features`.

## Facial Expression Feature Extraction

1. Download the FER-VT model weights [https://github.com/ZBigFish/FER-VT](https://github.com/ZBigFish/FER-VT) and place them at `/root/autodl-tmp/code/model`.
2. Run the following command to extract facial expression features:

   ```bash
   python extract_face_fervt.py
   ```

   The facial expression features are saved in `/root/autodl-tmp/features/extracted_features_without_xml`.

## Feature Fusion

To perform feature fusion, run:

   ```bash
   python cross_attention_without_xml.py
   ```

   The final results are saved after running the script.

# Cite
If you plan to apply or extend our work, please cite the following paper.
```bibtex
@inproceedings{wang2025towards,
  title={Towards patronizing and condescending language in chinese videos: A multimodal dataset and detector},
  author={Wang, Hongbo and Lu, Junyu and Han, Yan and Ma, Kai and Yang, Liang and Lin, Hongfei},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
# Poster (ICASSP 2025)
![Poster.](https://github.com/dut-laowang/PCLMM/blob/main/figure/ICASSP_poster_01.png)
