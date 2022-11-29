import os
from pylab import *


def best_model_path(exp_name):
    res_root = "results"
    checkpoint_root = os.path.join(res_root, exp_name, "checkpoints")
    checkpoint_files = os.listdir(checkpoint_root)
    checkpoint_files = [p for p in checkpoint_files if "epoch" in p]
    checkpoint_file = checkpoint_files[0]
    return os.path.join(checkpoint_root, checkpoint_file)
