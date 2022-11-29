import yaml
import pandas as pd
import os
import pdb

cfgs_root = 'cfgs'
cfg_list = os.listdir(cfgs_root)
cfg_list = sorted([p for p in cfg_list if 'yaml' in p])

skip_key = ['data_file','with_info_guide','similarity','resume','val_freq','save_top_k','num_sanity_val_steps','step','epoch']

all_data = []
for cfg_file in cfg_list:
    exp_name = cfg_file.split('.')[0]
    file_path = os.path.join(cfgs_root,cfg_file)
    x = {'exp':exp_name}
    with open(file_path,'r') as f:
        x.update(yaml.load(f,Loader=yaml.FullLoader))
    results_file = os.path.join('results',exp_name.upper(),'metrics.csv')
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        y = df.to_dict(orient='records')[0]
        x.update(y)
    x = {key:(','.join([str(p) for p in value]) if isinstance(value,list) else value) for (key,value) in x.items() if not key in skip_key}
    all_data.append(x)


df = pd.DataFrame(all_data)

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop, axis=1)

df.to_csv('exp_records.csv',index=False)