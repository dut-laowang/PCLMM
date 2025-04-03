# PCLMM
 The code implementation for the article "Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Framework". The paper is now **accepted by ICASSP 2025**, we will update the code in **early February, 2025**.

 The link for this paper in arxiv is https://arxiv.org/abs/2409.05005
 
 a) Data Collection. Refining annotation guidelines and gathering data from Bilibili. 
 
 b) PCLMM dataset. A high-quality annotated dataset with PCL frame spans. 
 
 c) MultiPCL detector. A cross-attention mechanism.
 
 ![Our framework for this paper.](https://github.com/dut-laowang/PCLMM/blob/main/figure/P8.PNG)

# Updation
**February 9, 2025** – 🔥 The [Annotation_Link.csv](https://github.com/dut-laowang/PCLMM/blob/main/data/Annotation_Link.csv) containing the original video links has been updated.

**April 3, 2025**– 🔥 We have released the complete process for calling the PCLMM code.

# Dataset

The PCLMM dataset can be downloaded at [https://zenodo.org/records/14840197](https://zenodo.org/records/14840197)

The PCLMM dataset is sourced from Bilibili, the largest online community for young people in China. Our work aims to uncover microaggressions targeted at vulnerable groups, including discriminatory and patronizing language expressions (715 annotated videos). 

You can download the dataset `Annotation_Link.csv` for detailed annotation.

The collection of this dataset follows Bilibili's Developer Agreement and Privacy Policy, with all data sourced from publicly available Bilibili video links. Please comply with Bilibili's relevant usage regulations when using this dataset to avoid any negative impact on the platform's services.

# Code

## Dataset
Download the PCLMM and place it at `/root/autodl-tmp/PCLMM`.

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

1. Download the BERT-Chinese model weights and place them at `/root/autodl-tmp/code/bert_chinese`.
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
@misc{wang2024patronizingcondescendinglanguagechinese,
      title={Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Detector}, 
      author={Hongbo Wang and Junyu Lu and Yan Han and Kai Ma and Liang Yang and Hongfei Lin},
      year={2024},
      eprint={2409.05005},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.05005}, 
}
```
