import h5py
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import random


class set(Dataset):
    def __init__(self, args, split, training):

        data_file = [
            "data/summe.h5",
            "data/tvsum.h5",
            "data/eccv16_dataset_ovp_google_pool5.h5",
            "data/eccv16_dataset_youtube_google_pool5.h5",
        ]

        self.feature_data = {}
        for file in data_file:
            file_name = file.split("/")[-1]
            if "_" in file_name:
                set_name = file_name.split("_")[2]
            else:
                set_name = file_name.split(".")[0]
            self.feature_data[set_name] = h5py.File(file, "r")

        if training:
            split = split * getattr(args, "set_multiply", 1)
        self.video_list = split
        self.eval_metric = "avg" if "tvsum" in args.split_file else "max"
        self.max_video = args.max_video

        self.training = training
        self.batch_size = args.batch_size_train

    def __len__(self):
        return len(self.video_list)

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
        vid_name = self.video_list[index]
        set_name = vid_name.split("_")[2]
        vid_id = vid_name.split("/")[-1]
        data = self.feature_data[set_name][vid_id]

        feature = torch.from_numpy(data["features"][...])
        gtscore = torch.from_numpy(data["gtscore"][...])
        if self.training:
            if not self.batch_size == 1:
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
        return inputs
