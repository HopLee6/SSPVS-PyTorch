from ntpath import join
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import importlib
import json


class DInterface(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.module_train = importlib.import_module(
            "datasets." + self.args.set_name_train
        )
        train_file = self.args.train_list
        with open(train_file, "r") as f:
            self.train_list = json.load(f)

        val_file = self.args.val_list
        with open(val_file, "r") as f:
            self.val_list = json.load(f)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.trainset = self.module_train.set(self.args, self.train_list)
            self.valset = self.module_train.set(self.args, self.val_list)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.args.batch_size_train,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.args.batch_size_val,
            num_workers=self.args.num_workers,
        )
