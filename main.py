import argparse
import os
import yaml
import json

import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from utils.general_utils import *


from datasets import DInterface
from models import MInterface


def main(args):
    seed_everything(args.seed, workers=True)

    exp_name = args.config.split("/")[-1].replace(".yaml", "").upper()

    split_file = args.split_file
    with open(split_file, "r") as f:
        split_data = json.load(f)
    logger = CSVLogger(save_dir="results", name="", version=exp_name)
    args.save_dir = logger.log_dir

    for split in split_data:
        data_module = DInterface(args, split)
        if args.resume:
            model = MInterface.load_from_checkpoint(
                checkpoint_path=args.resume, args=args, strict=False
            )
        else:
            model = MInterface(args)

        trainer = pl.Trainer(
            gpus=-1,
            max_epochs=args.max_epochs,
            checkpoint_callback=False,
            check_val_every_n_epoch=args.val_freq,
            logger=logger,
            accelerator="ddp_spawn",
            plugins=DDPPlugin(find_unused_parameters=False),
            num_sanity_val_steps=args.num_sanity_val_steps,
            deterministic=True,
        )
        trainer.fit(model, data_module)

    old = os.path.join("results", exp_name, "metrics.csv")
    new = old.replace("metrics", "metrics_final")
    rename(old, new)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        setattr(args, k, v)
    main(args)
