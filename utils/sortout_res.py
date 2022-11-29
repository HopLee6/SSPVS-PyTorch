import yaml
import pandas as pd
import os
import numpy as np
from functools import reduce
import sys


def get_best(df, best_f):
    rank = [df[:, i].argsort().argsort() for i in range(2)]
    if best_f:  # find best f-score
        rank = rank[0]
    else:  # considering both f-score and rank-based evaluation
        rank = reduce(lambda x, y: x + y, rank)
    return df[rank.argmax()]


cfgs_root = sys.argv[1].split("/")[0]

cfg_list = os.listdir(cfgs_root)
cfg_list = sorted([p for p in cfg_list if "yaml" in p])

show_key = [
    "kendalltau",
    "spearmanr",
    "exp",
]

best_f_case = ["sm_t", "ts"]

all_data = []
for cfg_file in cfg_list:
    exp_name = cfg_file.split(".")[0]
    results_file = os.path.join("results", exp_name.upper(), "metrics_final.csv")
    if os.path.exists(results_file):
        x = {"exp": exp_name}
        file_path = os.path.join(cfgs_root, cfg_file)

        with open(file_path, "r") as f:
            x.update(yaml.load(f, Loader=yaml.FullLoader))
        x["pretrain"] = True if x["resume"] else False
        x = {
            key: (
                ",".join([str(p) for p in value]) if isinstance(value, list) else value
            )
            for (key, value) in x.items()
            if key in show_key
        }

        df = pd.read_csv(results_file).to_numpy()[:, :3]
        n_split = 1 if "_T_" in results_file else 5
        epoch = df.shape[0] // n_split
        df_split = [df[epoch * i : epoch * i + epoch] for i in range(n_split)]
        best_list = []

        temp = [p in cfg_file for p in best_f_case]
        best_f = any(temp)

        for p in df_split:
            best_res = get_best(p, best_f)
            best_list.append(best_res)
        y = np.array(best_list).mean(0)
        y = np.around(y, 3)

        metrics = {"f_score": y[0], "kendalltau": y[1], "spearmanr": y[2]}
        x.update(metrics)
        all_data.append(x)

df = pd.DataFrame(all_data)

df.to_csv("records.csv", index=False)
