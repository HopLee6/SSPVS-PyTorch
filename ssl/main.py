import argparse
import os
import yaml

import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPPlugin

from datasets import DInterface
from models import MInterface
from utils.general_utils import best_model_path


def main(args):
    seed_everything(args.seed, workers=True)

    exp_name = args.config.split("/")[-1].replace(".yaml", "").upper()

    data_module = DInterface(args)

    # Trainer configuration
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor[0],
        mode=args.monitor[1],
        save_top_k=args.save_top_k,
        save_last=True,
        filename="{{{}}}-{{{}:.4f}}".format("epoch", args.monitor[0]),
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    if args.test:
        logger = CSVLogger(save_dir="results", name="", version=exp_name)
    else:
        logger = TensorBoardLogger(
            save_dir="results", name="", version=exp_name, default_hp_metric=False
        )
    args.save_dir = logger.log_dir

    if args.test:
        resume_path = best_model_path(exp_name) if not args.resume else args.resume
    else:
        resume_path = (
            os.path.join("results", exp_name, "checkpoints", "last.ckpt")
            if args.resume == "auto"
            else args.resume
        )

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=resume_path,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=args.val_freq,
        logger=logger,
        accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=False),
        num_sanity_val_steps=args.num_sanity_val_steps,
        deterministic=True,
    )

    if args.test:
        print("Loading checkpoint: {}".format(resume_path))
        model = MInterface.load_from_checkpoint(checkpoint_path=resume_path, args=args)
        trainer.test(model=model, datamodule=data_module)
        return

    model = MInterface(args)
    trainer.fit(model, data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        setattr(args, k, v)
    main(args)
