# CH-SIMS v2.0

Official codes for paper "Make Acoustic and Visual Cues Matter: CH-SIMS v2.0 Dataset and AV-Mixup Consistent Module" (ICMI 2022)

### [Dataset HomePage](https://thuiar.github.io/sims.github.io/chsims)

### Illustration of CH-SIMS v2.0 Data
<p align="center">
  <img width="800" src="show/CH-SIMSv2.0.png">
</p>

### Official Baselines Results
<p align="center">
  <img width="800" src="show/ModelResults.png">
</p>

### Data Download

1. CH-SIMS v2(s) - Supervised data:
  - [Google Drive](https://drive.google.com/drive/folders/1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA?usp=sharing)
  - [Baiduyun Drive]()

2. CH-SIMS v2(u) - Unsupervised data:
  - [Google Drive](https://drive.google.com/drive/folders/1llIbm3gwyJRwwk58RUQHWBNKjHI9vGGB?usp=sharing)
  - [Baiduyun Drive]()

### Data path

config/config.py --> modify parameter "root_dataset_dir" line 32 of your dataset path

### Run
If you want to run the AV-MC framework: 

```
python run.py --is_tune Flase --modelName v1
```

If you want to run the AV-MC(Semi) framework 

```
python run.py --is_tune Flase --modelName v1_semi
```
### Citation

If this paper is useful for your research, please cite us at: 

### Contact

For any questions, please email [Yihe Liu](mailto:512796310@qq.com) or [Ziqi Yuan](mailto:yzq21@mails.tsinghua.edu.cn)
