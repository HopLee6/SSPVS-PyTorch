import importlib
import torch
import pytorch_lightning as pl
from models.summary_module import summary_tool
from models import losses as loss_module


class MInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        video_embed_module = importlib.import_module(
            "models." + self.hparams.model_name
        )
        self.video_model = video_embed_module.model(self.hparams)
        self.loss = loss_module.loss(self.hparams)

    def forward(self, x):
        x.update(self.video_model(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        train_batch.update(self(train_batch))
        loss = self.loss(train_batch)
        return loss

    def validation_step(self, test_batch, batch_idx):
        test_batch.update(self.video_model(test_batch))
        test_batch.update(summary_tool(test_batch))
        metrics = {
            "f_score": test_batch["f_score"],
            "kendalltau": test_batch["kendalltau"],
            "spearmanr": test_batch["spearmanr"],
        }
        self.log_dict(metrics, prog_bar=False, sync_dist=True, logger=True)
        return metrics

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)
