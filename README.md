# Progressive Video Summarization via Multimodal Self-supervised Learning (SSPVS)
[Paper](https://arxiv.org/pdf/2201.02494.pdf) | [Supplementary Material](https://unimelbcloud-my.sharepoint.com/:b:/g/personal/haopengl1_student_unimelb_edu_au/EQnk1g8ZcE9BsGFJ-6upSXUBR5n1YVzeBjhfKglk9tJmNQ?e=P15I1c)

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


Download the pretrained model [pretrained_model.ckpt](https://figshare.com/articles/journal_contribution/SSPVS/29979121) to the root dictionary.

**OR**

Follow the following steps to train the self-supervised model.

### Data Preparation

Download the visual features [feat.z*](https://figshare.com/articles/journal_contribution/SSPVS/29979121) and text information embeddings [info_embed.zip](https://figshare.com/articles/journal_contribution/SSPVS/29979121) of the YTVT dataset and uncompress them to `ssl/features/` and `ssl/info_embed/`, respectively.


### Self-supervised Pretraining
Run the following command in `ssl/` to train the self-supervised model: 

```
$ CUDA_VISIBLE_DEVICES=0,1 python main_ssl.py --config ssl.yaml
```
The trained model is saved in `ssl/results/SSL/checkpoints/`.


## Progressive Video Summarization

### Data Preparation

Download the [data.zip](https://figshare.com/articles/journal_contribution/SSPVS/29979121) and uncompress it to ``data/``.


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


## Source Data

We provide the text information of YTVT in [YTVT.zip](https://figshare.com/articles/journal_contribution/SSPVS/29979121). Besides, we also provide the re-collected text information of SumMe and TVSum [SumMe_text_info.json/TVSum_text_info.json](https://figshare.com/articles/journal_contribution/SSPVS/29979121).

## License and Citation

The use of this code is RESTRICTED to **non-commercial research and educational purposes**.

If you use this code or reference our paper in your work please cite this publication as:

```
@inproceedings{li2023progressive,
  title={Progressive Video Summarization via Multimodal Self-supervised Learning},
  author={Li, Haopeng and Ke, Qiuhong and Gong, Mingming and Drummond, Tom},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5584--5593},
  year={2023}
}
```
## Acknowledgement

The code is developed based on [VASNet](https://github.com/ok1zjf/VASNet).

