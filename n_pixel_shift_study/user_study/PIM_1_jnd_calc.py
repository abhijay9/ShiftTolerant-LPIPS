import argparse
import tensorflow as tf
import numpy as np
import imageio
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import sys
sys.path.insert(0,'/pathToPIM1...') # Add path to PIM repo
from perceptual_quality import pim
import os
import glob
import time
import json
from tqdm import tqdm
import skimage.transform

def load_data(impath):
	img = imageio.imread(impath)
	l = np.array(img)[:,0:256,:]
	r = np.array(img)[:,256+20:,:]
	l = l[:,:,0:3].astype(np.float32) / 255.0
	r = r[:,:,0:3].astype(np.float32) / 255.0
	l = np.expand_dims(np.array(l), axis=0)
	r = np.expand_dims(np.array(r), axis=0)
	return l, r

metric = pim.load_trained("pim-1")

f = open("./user_jnd.json","rb")
responsesMoreThan2 = json.load(f)

ds = []
sames = []
for k in tqdm(responsesMoreThan2.keys()):
    im = "/".join(["./0_9_pix_shift",k])
    resp = np.mean(responsesMoreThan2[k])
    sames.append(resp)
    l, r = load_data(im)
    ds.append(metric((l,r)))

sames = np.array(sames)
ds = np.array(ds)

with open('pim1_user_jnd_ds.npy', 'wb') as f:
    np.save(f, ds)
with open('pim1_user_jnd_sames.npy', 'wb') as f:
    np.save(f, sames)