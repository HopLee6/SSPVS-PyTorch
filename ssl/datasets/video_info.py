import os
import random

import h5py
from random import randrange
import torch
from torch.utils.data import Dataset
import torch
from pytorch_transformers import *


class set(Dataset):
    def __init__(self, args, split):
        self.feature_root = args.feature_root
        self.info_embed_root = args.info_embed_root
        self.max_cat = args.max_cat
        self.max_query = args.max_query
        self.max_title = args.max_title
        self.max_desc = args.max_desc
        self.max_video = args.max_video

        self.video_list = split * 8

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.pad_id = self.tokenizer.encode("[PAD]")[0]
        self.sep_id = torch.tensor(self.tokenizer.encode("[SEP]"))
        self.cls_id = torch.tensor(self.tokenizer.encode("[CLS]"))

        self.text_type = getattr(args, "text_type", "all")

        if not self.text_type == "all":

            map_dict = {
                "cat": self.max_cat,
                "query": self.max_query,
                "title": self.max_title,
                "description": self.max_desc,
            }

            self.max_text = map_dict[self.text_type]

    def __len__(self):
        return len(self.video_list)

    def pad_info(self, tensor, max_len):
        seq_len = tensor.size(0)
        if seq_len >= max_len:
            start = random.randint(0, seq_len - max_len)
            return tensor[start : start + max_len]
        else:
            container = torch.ones(max_len) * self.pad_id
            container[:seq_len] = tensor
            return container

    def process_info_embed(self, vid_id):
        info_embed_path = os.path.join(self.info_embed_root, vid_id + ".h5")

        if self.text_type == "all":

            with h5py.File(info_embed_path, "r") as f:
                cat_ids = torch.tensor(f["cat"][...])
                query_ids = torch.tensor(f["query"][...])
                title_ids = torch.tensor(f["title"][...])
                desc_ids = torch.tensor(f["description"][...])

            cat_ids_pad = self.pad_info(cat_ids, self.max_cat)
            query_ids_pad = self.pad_info(query_ids, self.max_query)
            title_ids_pad = self.pad_info(title_ids, self.max_title)
            description_ids_pad = self.pad_info(desc_ids, self.max_desc)

            info_input = torch.cat(
                [
                    self.cls_id,
                    cat_ids_pad,
                    self.sep_id,
                    query_ids_pad,
                    self.sep_id,
                    title_ids_pad,
                    self.sep_id,
                    description_ids_pad,
                    self.sep_id,
                ]
            ).long()
            return info_input
        else:
            with h5py.File(info_embed_path, "r") as f:
                info_embed = torch.tensor(f[self.text_type][...])
            info_embed_pad = self.pad_info(info_embed, self.max_desc)

            info_input = torch.cat(
                [
                    self.cls_id,
                    info_embed_pad,
                    self.sep_id,
                ]
            ).long()
            return info_input

    def pad_vid(self, feature):
        length, dim = feature.size()
        if length >= self.max_video:
            start = random.randint(0, length - self.max_video)
            feature_input = feature[start : start + self.max_video]
        else:
            feature_input = torch.zeros(self.max_video, dim)
            feature_input[0:length] = feature
        return feature_input

    def __getitem__(self, index):

        vid_id = self.video_list[index]
        feature_path = os.path.join(self.feature_root, vid_id + ".h5")
        with h5py.File(feature_path, "r") as f:
            feature = torch.from_numpy(f["features"][...])

        feature = self.pad_vid(feature)

        masked_pos = randrange(self.max_video)

        masked_feature = feature[masked_pos]

        if random.random() > 0.5:
            vid_id_ = vid_id
            label = 1
        else:
            while True:
                vid_id_ = random.choice(self.video_list)
                if not vid_id_ == vid_id:
                    break
            label = 0
        info_embed = self.process_info_embed(vid_id_)

        return {
            "video_feature": feature,
            "info_embed": info_embed,
            "masked_feature": masked_feature,
            "masked_pos": masked_pos,
            "label": label,
        }
