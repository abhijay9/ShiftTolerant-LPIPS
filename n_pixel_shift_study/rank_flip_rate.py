import pandas as pd
import numpy as np
import argparse
import sys
import os
from glob import glob
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--rrf_results', type=str, default='', help='')
opt = parser.parse_args()

datasets = ['cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional']
total_test_instances = [4720, 4720, 9440, 1888, 10856, 4720]

test_instances_rank_flipped = pd.read_csv(opt.rrf_results, header=None).drop_duplicates(0)[1].to_list()
print('test instances rank flipped: ', test_instances_rank_flipped)
perc_rrf = []
for d,n,t in zip(datasets,test_instances_rank_flipped,total_test_instances):
    v = 100*n/t
    perc_rrf.append(v)
    print('%s : %f'%(d,v))

print('Result =====> %d pix shift mean r_rf: %f (final val) \n\nExtra'%(int(opt.rrf_results.split("_n-")[1][0]),np.mean(perc_rrf)))
print('sum of rank flips: %d'%(np.sum(test_instances_rank_flipped)))
print('100*sum_of_rank_flips/num_test_instances: %f'%(100*np.sum(test_instances_rank_flipped)/np.sum(total_test_instances)))