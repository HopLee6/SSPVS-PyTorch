import pytorch_lightning as pl
from torch.utils.data import DataLoader
import importlib


class DInterface(pl.LightningDataModule):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.module_ = importlib.import_module("datasets." + self.args.set_name_test)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.trainset = self.module_.set(
                self.args, self.split["train_keys"], training=True
            )
            self.valset = self.module_.set(
                self.args, self.split["test_keys"], training=False
            )

        if stage == "test" or stage is None:
            self.testset = self.module_.set(
                self.args,
                self.split["test_keys"] + self.split["train_keys"],
                training=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.args.batch_size_train,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.args.batch_size_val, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.args.batch_size_val, num_workers=0
        )
