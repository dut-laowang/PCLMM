# PCLMM
 The code implementation for the article "Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Framework". The paper is currently under review for **ICASSP 2025**.

 The link for this paper in arxiv is https://arxiv.org/abs/2409.05005
 
 a) Data Collection. Refining annotation guidelines and gathering data from Bilibili. 
 
 b) PCLMM dataset. A high-quality annotated dataset with PCL frame spans. 
 
 c) MultiPCL detector. A cross-attention mechanism.
 
 ![Our framework for this paper.](https://github.com/dut-laowang/PCLMM/blob/main/figure/P8.PNG)
 
# Dataset
The PCLMM dataset can be downloaded at https://doi.org/10.5281/zenodo.13710863

PCLMM - 715 annotated videos from chinese platform Bilibili 

You can download the dataset `Annotation.csv` for detailed annotation.
# Code
The main code has been open-sourced. The video modality feature extraction is in `video_extract`, audio modality feature extraction is in `audio_extract`, facial expression feature extraction is in `face_extract`, and text is in `text_extract`. The multimodal fusion uses MHCA in the `model` folder. 
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
# Updating
The code will be updated after the article is accepted.
