# Progressive Video Summarization via Multimodal Self-supervised Learning (SSPVS)
 [Paper](https://arxiv.org/pdf/2201.02494.pdf)

Haopeng Li, Qiuhong Ke, [Mingming Gong](https://mingming-gong.github.io/), Tom Drummond

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023


## Introduction


Modern video summarization methods are based on deep neural networks that require a large amount of annotated data for training. However, existing datasets for video summarization are small-scale, easily leading to over-fitting of the deep models. Considering that the annotation of large-scale datasets is time-consuming, we propose a multimodal self-supervised learning framework to obtain semantic representations of videos, which benefits the video summarization task. Specifically, the self-supervised learning is conducted by exploring the  semantic consistency between  the videos and text in both coarse-grained and fine-grained fashions, as well as recovering  masked frames in the  videos. The multimodal framework is trained on a newly-collected dataset that consists of video-text pairs. Additionally, we introduce a progressive video summarization method, where the important content in a video is pinpointed progressively to generate better summaries. Extensive experiments have proved the effectiveness and superiority of our method in rank correlation coefficients and F-score.


## Requirements and Dependencies

- python=3.8.5
- pytorch=1.12, ortools=9.3.10497
- pylorch-lightning=1.6.5 
- pytorch-transformers=1.2.0


## Data Preparation

Download [data.zip](https://drive.google.com/file/d/1txVUTZNWDxXVGZUAOs7Hh7FqDLEOrs8w/view?usp=sharing) and uncompress it to ``data/``.

## Self-supervised Pretrained Model

Download [the pretrained model](https://drive.google.com/file/d/1VUSqlXuDZt0HW2TXv5bVl8jqfIjg4Wvx/view?usp=sharing) to the root dictionary.


## Training and Evaluation
Run the following command to train the model: 

```
$ sh main.sh CFG_File
```

Example for training the model on TVSum in the canonical setting: 

```
$ sh main.sh cfgs/sm_a.yaml
```

The results are recoded in ``records.csv``.




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
<!-- ## Acknowledgement

The code is developed based on [VASNet](https://github.com/ok1zjf/VASNet). -->

