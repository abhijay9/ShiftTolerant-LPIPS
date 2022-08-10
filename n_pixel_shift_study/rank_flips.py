"""
Reads the d0 and d1 predicted for each sample from the csv stored in the 'results' folder. Then, it computes 2afc score (stored in 2afcScores folder in evaluations) and the rank flip rate r_rf (stored in rankFlips folder in evaluations).
"""
import pandas as pd
import numpy as np
import argparse
import sys
import os
from glob import glob
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--shifted_0_results', type=str, default='', help='')
parser.add_argument('--shifted_n_results', type=str, default='', help='')
# parser.add_argument('--outfile', type=str, default='', help='')
parser.add_argument('--type', type=str, default='seq', help='seq,pair')
parser.add_argument('--model_name', type=str, default='pieapp', help='pieapp, elpips or model_name..')
opt = parser.parse_args()

with open('util/config/global_config.json') as f:
    gl_conf = json.load(f)
gtFilePath = gl_conf['2afc_rootdir']+'/2afc/val'

print(opt.shifted_n_results.split('_'))
num_pixels_shifted = opt.shifted_n_results.split('_')[-1][-2]

print(opt.model_name)
if opt.type == 'seq':
    df = pd.read_csv(opt.shifted_0_results, header=None)
    dfShift = pd.read_csv(opt.shifted_n_results, header=None)

    df.columns = ['distortion','p_index','img_index',opt.model_name]
    dfShift.columns = ['distortion','p_index','img_index',opt.model_name]

datasets = ['cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional']

counts = []
countsBetter = []
shift_0_2AFCScores = []
shift_n_2AFCScores = []

if opt.type == 'pair':
    shift_0_filepaths = glob(opt.shifted_0_results)
    shift_n_filepaths = glob(opt.shifted_n_results)
    shift_n_filepaths.sort()
    shift_0_filepaths.sort()

for d_indx, dataset in enumerate(datasets):
    count = 0
    countBetter = 0

    if opt.type == 'seq':
        df_dataset = df[df['distortion']==dataset]
        # print(df_dataset[df_dataset['p_index']=='p0'][opt.model_name])
        d0s = np.array(df_dataset[df_dataset['p_index']=='p0'][opt.model_name].tolist())
        d1s = np.array(df_dataset[df_dataset['p_index']=='p1'][opt.model_name].tolist())

        df_dataset = dfShift[dfShift['distortion']==dataset]
        d0sShift = np.array(df_dataset[df_dataset['p_index']=='p0'][opt.model_name].tolist())
        d1sShift = np.array(df_dataset[df_dataset['p_index']=='p1'][opt.model_name].tolist())
    else:
        shift_0_filepath = shift_0_filepaths[d_indx]
        shift_n_filepath = shift_n_filepaths[d_indx]
        print(dataset, shift_0_filepath, shift_n_filepath)
        
        df = pd.read_csv(shift_0_filepath, header=None)
        dfShift = pd.read_csv(shift_n_filepath, header=None)
        
        df.columns = ['img_index','d0','d1','d0Acc','d1Acc']
        dfShift.columns = ['img_index','d0','d1','d0Acc','d1Acc']
        d0s = np.array(df['d0'].tolist())
        d1s = np.array(df['d1'].tolist())
        d0sShift = np.array(dfShift['d0'].tolist())
        d1sShift = np.array(dfShift['d1'].tolist())

    gts = []
    gt_files = glob(os.path.join(gtFilePath,dataset,'judge/*'))
    gt_files.sort()
    for file in gt_files:
        gts.append(np.load(file))
    gts=np.array(gts)

    if opt.model_name == 'MSSSIM':
        d0s = 1.-d0s
        d1s = 1.-d1s
        d0sShift = 1.-d0sShift
        d1sShift = 1.-d1sShift

    d0LTd1 = (d0s<d1s)
    d0LTd1Shift = (d0sShift<d1sShift)

    d0GTd1 = (d0s>d1s)
    d0GTd1Shift = (d0sShift>d1sShift)
    oneMinGts = (1.-gts).flatten()
    Gts = (gts).flatten()
    d0EQd1 = d1s==d0s
    d0EQd1Shift = d1sShift==d0sShift

    scores = (d0s<d1s)*(1.-gts).flatten() + (d1s<d0s)*gts.flatten() + (d1s==d0s)*.5
    shift_0_2AFCScores.append(np.mean(scores))
    
    scoresShift = (d0sShift<d1sShift)*(1.-gts).flatten() + (d1sShift<d0sShift)*gts.flatten() + (d1sShift==d0sShift)*.5
    shift_n_2AFCScores.append(np.mean(scoresShift))
    # print(np.mean(scoresShift))

    if opt.type == 'seq':
        img_indices = df_dataset[df_dataset['p_index']=='p0']['img_index'].tolist()
    else:
        img_indices = df['img_index'].tolist()

    for i, img_index in enumerate(img_indices):
        if d0LTd1[i] != d0LTd1Shift[i]:
            count+=1
            scoreOrig = d0LTd1[i]*oneMinGts[i] + d0GTd1[i]*Gts[i] + d0EQd1[i]*.5
            scoreShift = d0LTd1Shift[i]*oneMinGts[i] + d0GTd1Shift[i]*Gts[i] + d0EQd1Shift[i]*.5
            if scoreShift>scoreOrig:
                countBetter+=1

    counts.append(str(count))
    countsBetter.append(str(countBetter))

outPath = 'n_pixel_shift_study/evaluations/2afcScores/'
with open(outPath+opt.model_name+'_n-'+str(num_pixels_shifted)+'_2afc.csv','w') as f:
    f.write('dataset,noShift,shifted\n')
    for d_indx, dataset in enumerate(datasets):
        f.write(dataset+','+str(shift_0_2AFCScores[d_indx])+','+str(shift_n_2AFCScores[d_indx])+'\n')
f.close()

outPath = 'n_pixel_shift_study/evaluations/rankFlips/'
with open(outPath+opt.model_name+'_n-'+str(num_pixels_shifted)+'_rrf.csv','w') as f:
    for d_indx, dataset in enumerate(datasets):
        f.write(dataset+','+str(counts[d_indx])+'\n')
        f.write(dataset+','+str(countsBetter[d_indx])+'\n')
f.close()
