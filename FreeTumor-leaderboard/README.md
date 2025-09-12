<div align="center">
<h1>FreeTumor's performance on public leaderboard</h1>

</div>

This repo present the evaluation codes, models and results of FreeTumor on several public leaderboard, i.e., [FLARE25](https://www.codabench.org/competitions/7149/\#/results-tab), [FLARE23](https://codalab.lisn.upsaclay.fr/competitions/12239\#results), and [KiTS](https://kits19.grand-challenge.org/evaluation/challenge/leaderboard/). 

For the training codes, it is the same as the [FreeTumor-Abdomen](https://github.com/Luffy03/FreeTumor/tree/main/FreeTumor-Abdomen) repo, and the training implementation can be found in [README](https://github.com/Luffy03/FreeTumor). 

**Note that for FLARE25 pan-cancer dataset, we train one model for all three abdomen tumor types, i.e., liver, pancreas, and kidney tumors. For KiTS, we train on only kidney tumors.**

## Evaluation

To evaluate our models, please download the following checkpoints. We also provide the prediction results for reference.

| Leaderboard                                                                 | Result                                                                                             |  Checkpoint                                                                                       |
| :-------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------| :----------------------------------------------------------------------------------------------- |
| [FLARE25](https://www.codabench.org/competitions/7149/#/results-tab)        | [51.0](https://drive.google.com/file/d/1GceUiNST7JMM5k66GaV5farzPbalqc7w/view?usp=sharing)         | [Download](https://drive.google.com/file/d/1Qi4Ms4dSyQc0AxemGNKlsaaDAcTaRoaY/view?usp=sharing)    |
| [FLARE23](https://codalab.lisn.upsaclay.fr/competitions/12239#results)      | [69.8(devel.)](https://drive.google.com/file/d/1QSzBCghQyLG-01i4nrXXOAzhOOVenSen/view?usp=sharing) |  [Download](https://drive.google.com/file/d/1Qi4Ms4dSyQc0AxemGNKlsaaDAcTaRoaY/view?usp=sharing)   |
| [FLARE23](https://codalab.lisn.upsaclay.fr/competitions/12239#results)      | [64.9(final)](https://drive.google.com/file/d/1qSswI5GBJFySlL6dOFGi5jHL-GDPiNoW/view?usp=sharing)  |  [Download](https://drive.google.com/file/d/1Qi4Ms4dSyQc0AxemGNKlsaaDAcTaRoaY/view?usp=sharing)   |
| [KiTS](https://kits19.grand-challenge.org/evaluation/challenge/leaderboard/) | [92.6](https://drive.google.com/file/d/1JBXhQ8136g4sW_qmVoY19CJD_PgMhKN3/view?usp=sharing)         | [Download](https://drive.google.com/file/d/1OtcCleQMLkl52odjqWQsHEOhmAGUhJrf/view?usp=sharing)   |

To evaluate on the dataset, you can easily run [evaluate.py](./evaluate.py) as follows:
```
python evaluate.py --test_data_path $YOUR_PATH_TO_DATA --save_prediction_path $YOUR_PATH_TO_SAVE_PREDICTION --trained_pth $YOUR_PATH_TO_CHECKPOINT
```

## Training

### Datasets

The labels of training datasets can be found at [Hugging face](https://huggingface.co/datasets/Luffy503/FreeTumor), which contain organ labels for tumor synthesis. You can download the images of datasets from their original sources or our previous work [Large-Scale-Medical](https://github.com/Luffy03/Large-Scale-Medical). Specifically, the organ labels are defined as:
```
# abdomen
0: background
1: liver
2: liver tumors
3: pancreas
4: pancreas tumors
5: kidney
6: kidney tumors
```

The path of datasets should be organized as:
```
├── /data/FreeTumor
    ├── Dataset003_Liver
        ├──imagesTr
        └──labelsTr
    ├── Dataset007_Pancreas
    └── ...
```

### Implementations

First, you need to train a baseline segmentation model as the discriminator for synthesis training (or you can download ours). The baseline segmentation model is placed in './baseline/'
```
├── baseline
    └── model_baseline_segmentor.pt ### for abdomen, 7 output channels
```

The synthesis training is conducted on 8*H800 GPUs while the segmentation training can be done with one 3090 GPU. Simple commands for training:
```
# Synthesis training
sh Syn_train.sh

# Segmentation training
sh Free_train.sh
```

Notably, currently we provide codes to train a generalist model, which can synthesize liver tumors, pancreas tumors, and kidney tumors (output by different channels). If you want to train specialist models for specific types of tumors (e.g., one model for liver tumors and another model for pancreas tumors), you need to check the codes as [here](https://github.com/Luffy03/FreeTumor/blob/main/FreeTumor-Chest/models/TumorGAN.py) and modify the labels as follows:
```
0: background
1: organ
2: tumor/lesion
```

## Acknowledgement
 
This work is highly inspired by series of [pioneering works](https://github.com/MrGiovanni/SyntheticTumors) led by Prof. [Zongwei Zhou](https://scholar.google.com/citations?user=JVOeczAAAAAJ&hl=en). **We highly appreciate their great efforts.**

## Citation

If you find our codes useful, please consider to leave a star and cite our paper as follows, we would be highly grateful (^o^)/. 

In addition, some previous papers that contributed to our work are listed for reference.

```bibtex
@article{wu2025freetumor,
  title={FreeTumor: Large-Scale Generative Tumor Synthesis in Computed Tomography Images for Improving Tumor Recognition},
  author={Wu, Linshan and Zhuang, Jiaxin and Zhou, Yanning and He, Sunan and Ma, Jiabo and Luo, Luyang and Wang, Xi and Ni, Xuefeng and Zhong, Xiaoling and Wu, Mingxiang and others},
  journal={arXiv preprint arXiv:2502.18519},
  year={2025}
}
@inproceedings{hu2023label,
  title={Label-free liver tumor segmentation},
  author={Hu, Qixin and Chen, Yixiong and Xiao, Junfei and Sun, Shuwen and Chen, Jieneng and Yuille, Alan L and Zhou, Zongwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7422--7432},
  year={2023}
}
@inproceedings{chen2024towards,
  title={Towards generalizable tumor synthesis},
  author={Chen, Qi and Chen, Xiaoxi and Song, Haorui and Xiong, Zhiwei and Yuille, Alan and Wei, Chen and Zhou, Zongwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11147--11158},
  year={2024}
}
```
