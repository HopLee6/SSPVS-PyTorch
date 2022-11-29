import os
import re
import pdb
from pylab import *

def best_model_path(exp_name):
    res_root = 'results'
    checkpoint_root = os.path.join(res_root,exp_name,'checkpoints')
    checkpoint_files = os.listdir(checkpoint_root)
    checkpoint_files = [p for p in checkpoint_files if 'epoch' in p]
    checkpoint_file = checkpoint_files[0]
    return os.path.join(checkpoint_root,checkpoint_file)

def vis_score(inputs,save_dir):
    save_path = os.path.join(save_dir,'vis_score')
    os.makedirs(save_path,exist_ok=True)
    machine_score = inputs['machine_score'][::15]
    machine_score -= machine_score.min()
    machine_score /= machine_score.max()

    gt_score = inputs['gtscore_up'][::15]
    gt_score -= gt_score.min()
    gt_score /= gt_score.max()

    # pdb.set_trace()
    video_id = inputs['video_id'][0]
    x_axis_data = list(range(len(machine_score)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_axis_data, machine_score, 'r')
    ax.plot(x_axis_data, gt_score, 'b')
    ax.set_aspect(aspect=100)
    fig.savefig(os.path.join(save_path,video_id+'.jpg'))
    plt.close()
