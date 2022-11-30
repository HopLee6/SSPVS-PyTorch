# Progressive Video Summarization via Multimodal Self-supervised Learning (SSPVS)
 [Paper](https://arxiv.org/pdf/2201.02494.pdf)

Haopeng Li, Qiuhong Ke, [Mingming Gong](https://mingming-gong.github.io/), Tom Drummond

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023


## Introduction

We propose a multimodal self-supervised learning framework to obtain semantic representations of videos, which benefits the video summarization task. 

Specifically, the self-supervised learning is conducted by exploring the  semantic consistency between  the videos and text in both coarse-grained and fine-grained fashions, as well as recovering  masked frames in the  videos. 

The multimodal framework is trained on a newly-collected dataset that consists of video-text pairs. 

Additionally, we introduce a progressive video summarization method, where the important content in a video is pinpointed progressively to generate better summaries.


## Requirements and Dependencies

- python=3.8.13
- pytorch=1.12, ortools=9.3.10497
- pytorch-lightning=1.6.5 
- pytorch-transformers=1.2.0


## Self-supervised Pretraining


Download the [pretrained model](https://unimelbcloud-my.sharepoint.com/:u:/g/personal/haopengl1_student_unimelb_edu_au/EW4hDPJnGWZHr8c19gZcXYQB2ajRX8bpdn4_c_SBfZ-Uig?e=xvhit3) to the root dictionary.

**OR**

Follow the following steps to train the self-supervised model.

### Data Preparation

Download the [visual features](https://unimelbcloud-my.sharepoint.com/:u:/g/personal/haopengl1_student_unimelb_edu_au/Efr65A_gDpdIqMxnxRWbIt4BpBe8XYhc4_KX2_QlhnyCig?e=znqRT2) and [text information embeddings](https://unimelbcloud-my.sharepoint.com/:u:/g/personal/haopengl1_student_unimelb_edu_au/EVlgG9lOExFNl3Ds1eBigdkBSqTDv7CR9e4vXKcpl_f3mQ?e=dcSdVs) of the YTVT dataset and uncompress them to `ssl/features/` and `ssl/info_embed/`, respectively.


### Self-supervised Pretraining
Run the following command in `ssl/` to train the self-supervised model: 

```
$ CUDA_VISIBLE_DEVICES=0,1 python main_ssl.py --config ssl.yaml
```
The trained model is saved in `ssl/results/SSL/checkpoints/`.


## Progressive Video Summarization

### Data Preparation

Download the [data](https://unimelbcloud-my.sharepoint.com/:u:/g/personal/haopengl1_student_unimelb_edu_au/ER72XF7I-_NBoGFpBghHSdEBAO753RSF6_cYTLvfMTlVXw?e=NLjcUO) and uncompress it to ``data/``.


### Training and Evaluation of Video Summarization
Run the following command in the root dictionary to train the video summarization model: 

```
$ sh main.sh CFG_FILE
```
where `CFG_FILE` is a configuration file (`*.yaml`) for different settings. We provide several configuration files in `cfgs/`. Here is an example for training the model on SumMe in the augmented setting: 

```
$ sh main.sh cfgs/sm_a.yaml
```

If you pretrain the model yourself, change `resume` in `CFG_FILE`  to the model saved in `ssl/results/SSL/checkpoints/`. The results of video summarization are recoded in ``records.csv``.


## YTVT Dataset

We also provide the original videos and text information of YTVT [here](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/haopengl1_student_unimelb_edu_au/End9OkDNT5FMixpK1LECLA8BwyEuaVKq4gD9nZVngQ4Eqg?e=DHrre3).

## License and Citation

The use of this code is RESTRICTED to **non-commercial research and educational purposes**.

If you use this code or reference our paper in your work please cite this publication as:

```
@article{haopeng2022video,
  title={Progressive Video Summarization via Multimodal Self-supervised Learning},
  author={Haopeng, Li and Qiuhong, Ke and Mingming, Gong and Drummond, Tom},
  journal={arXiv preprint arXiv:2201.02494},
  year={2022}
}
```
## Acknowledgement

The code is developed based on [VASNet](https://github.com/ok1zjf/VASNet).

