import pandas as pd
import numpy as np
import argparse
import sys
import os
from glob import glob
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--results_folder', type=str, default='', help='')
opt = parser.parse_args()

with open('util/config/global_config.json') as f:
    gl_conf = json.load(f)
gtFilePath = gl_conf['2afc_rootdir']+'/2afc/val'

datasets = ['cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional']

twoAFCScores = []

filepaths = glob(opt.results_folder+"/twoAFC*")
filepaths.sort()

for d_indx, dataset in enumerate(datasets):

    filepath = filepaths[d_indx]
    df = pd.read_csv(filepath, header=None)
    df.columns = ['img_index','d0','d1','d0Acc','d1Acc']
    d0s = np.array(df['d0'].tolist())
    d1s = np.array(df['d1'].tolist())

    if opt.results_folder.split('/')[1] == "MSSSIM":
        d0s = 1.-np.array(d0s)
        d1s = 1.-np.array(d1s)

    gts = []
    gt_files = glob(os.path.join(gtFilePath,dataset,'judge/*'))
    gt_files.sort()
    for file in gt_files:
        gts.append(np.load(file))
    gts=np.array(gts)

    d0LTd1 = (d0s<d1s)
    d0GTd1 = (d0s>d1s)

    scores = (d0s<d1s)*(1.-gts).flatten() + (d1s<d0s)*gts.flatten() + (d1s==d0s)*.5
    twoAFCScores.append(np.mean(scores))
    print(dataset,":",100*np.mean(scores))
print("Result =====> mean twoAFC:",100*np.mean(twoAFCScores))
