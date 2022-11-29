from math import e
import os

import h5py
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import random


class set(Dataset):
    def __init__(self, args, split, training):

        self.feature_data = h5py.File(args.data_file, "r")
        self.info_embed_root = args.info_embed_root_test

        if training:
            split = split * getattr(args, "set_multiply", 1)
        self.video_list = split
        self.eval_metric = "avg" if "tvsum" in args.data_file else "max"
        self.with_info_guide = args.with_info_guide
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.pad_id = self.tokenizer.encode("[PAD]")[0]
        self.sep_id = torch.tensor(self.tokenizer.encode("[SEP]"))
        self.cls_id = torch.tensor(self.tokenizer.encode("[CLS]"))

        self.max_cat = args.max_cat
        self.max_query = args.max_query
        self.max_title = args.max_title
        self.max_desc = args.max_desc
        self.max_video = args.max_video

        self.training = training

    def __len__(self):
        return len(self.video_list)

    def pad_info(self, tensor, max_len):
        seq_len = tensor.size(0)
        if seq_len >= max_len:
            return tensor[:max_len]
        else:
            container = torch.ones(max_len) * self.pad_id
            container[:seq_len] = tensor
            return container

    def process_info_embed(self, vid_id):
        info_embed_path = os.path.join(self.info_embed_root, vid_id + ".h5")
        with h5py.File(info_embed_path, "r") as f:
            cat_ids = torch.tensor(f["cat"][...])
            query_ids = torch.tensor(f["query"][...])
            title_ids = torch.tensor(f["title"][...])

        if self.training:
            cat_ids = self.pad_info(cat_ids, self.max_cat)
            query_ids = self.pad_info(query_ids, self.max_query)
            title_ids = self.pad_info(title_ids, self.max_title)

        info_input = torch.cat(
            [
                self.cls_id,
                cat_ids,
                self.sep_id,
                query_ids,
                self.sep_id,
                title_ids,
                self.sep_id,
            ]
        ).long()
        return info_input

    def pad_vid(self, feature, gt):
        length, dim = feature.size()
        if length >= self.max_video:
            start = random.randint(0, length - self.max_video)
            feature_input = feature[start : start + self.max_video]
            gt_input = gt[start : start + self.max_video]
        else:
            feature_input = torch.zeros(self.max_video, dim)
            gt_input = torch.zeros(self.max_video)
            feature_input[0:length] = feature
            gt_input[0:length] = gt
        return feature_input, gt_input

    def __getitem__(self, index):
        vid_id = self.video_list[index]
        data = self.feature_data[vid_id]

        feature = torch.from_numpy(data["features"][...])
        gtscore = torch.from_numpy(data["gtscore"][...])
        if self.training:
            feature, gtscore = self.pad_vid(feature, gtscore)
            inputs = {
                "video_feature": feature,
                "gtscore": gtscore,
            }
        else:
            inputs = {
                "video_feature": feature,
                "video_id": vid_id,
                "change_points": data["change_points"][...],
                "n_frames": data["n_frames"][()],
                "n_frame_per_seg": data["n_frame_per_seg"][...].tolist(),
                "picks": data["picks"][...],
                "user_summary": data["user_summary"][...],
                "user_scores": data["user_scores"][...],
                "gtscore": gtscore,
                "eval_metric": self.eval_metric,
            }
        if self.with_info_guide:
            inputs["info_ids"] = self.process_info_embed(vid_id)

        return inputs
