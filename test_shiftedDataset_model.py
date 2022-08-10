import numpy as np
import stlpips
from data import data_loader as dl
import argparse
from IPython import embed
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_mode', type=str, default='2afc', help='[2afc,jnd]')
parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn','val/superres','val/deblur','val/color','val/frameinterp'], help='datasets to test - for jnd mode: [val/traditional],[val/cnn]; for 2afc mode: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
parser.add_argument('--variant', type=str, default='', help='[], [vanilla], [antialiased] or [shift_tolerant] for network architecture variant')
parser.add_argument('--batch_size', type=int, default=50, help='batch size to test image patches in')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')
parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')
parser.add_argument('--test_type', type=str, default='twoAFC', help='[twoAFC], or [xshifted] for the type of tests: twoAFC (original) or xshifted (1,2,3.. pixel shift)')
parser.add_argument('--shift_direction', type=str, default='right', help='currently only testing for right')
parser.add_argument('--shift_type', type=str, default='crop', help='currently only testing for crop shift')
parser.add_argument('--num_pixels_shifted', type=int, default=0, help='0,1,2,3 pixel shift if shift_offset is 3')
parser.add_argument('--shift_offset', type=int, default=3, help='offset for crop-shifted test image 3 or 4')
parser.add_argument('--load_size', type=int, default=64, help='patch size for experiment')
opt = parser.parse_args()

with open('util/config/global_config.json') as f:
    gl_conf = json.load(f)
    
def data_to_gpu(data):
    data['ref'] = data['ref'].to(device=opt.gpu_ids[0])
    data['p0'] = data['p0'].to(device=opt.gpu_ids[0])
    data['p1'] = data['p1'].to(device=opt.gpu_ids[0])
    return data['ref'], data['p0'], data['p1']

def get_shifted_pair(distorted_img, ref_img, shift_direction='right', shift_type='crop', n=0, shift_offset=3):
    img_len = distorted_img.size()[-1]
    if shift_direction == 'right':
        if shift_type == 'crop':
            distorted_img = distorted_img[:, :, :, n:(img_len - shift_offset + n)]
            ref_img = ref_img[:, :, :, 0:(img_len - shift_offset)]
    return distorted_img, ref_img

def score_2afc_dataset(data_loader, func, name='', model_name='', save_dir=''):

    d0s = []
    d1s = []
    gts = []
    p0_paths = []

    for data in tqdm(data_loader.load_data(), desc=name):
        if opt.use_gpu:
            data['ref'], data['p0'], data['p1'] = data_to_gpu(data)
        d0s += func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s += func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts += data['judge'].cpu().numpy().flatten().tolist()
        p0_paths += data['p0_path']

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    
    if (model_name == 'MSSSIM'):
        d0s_msssim = 1.-np.array(d0s)
        d1s_msssim = 1.-np.array(d1s)
        scores = (d0s_msssim<d1s_msssim)*(1.-gts) + (d1s_msssim<d0s_msssim)*gts + (d1s_msssim==d0s_msssim)*.5
    else:
        scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    np.savetxt(save_dir + "/twoAFC"
                + "_model-" + model_name
                + "_distortion-" + name.split("/")[-1]
                + ".csv", np.array([p0_paths, d0s, d1s, scores, gts], dtype='object').T, delimiter=",", fmt='%s')

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_shifted_2afc_dataset(data_loader, func, name='', model_name='', test_type='', shift_direction='', shift_type='', n=1, save_dir='', shift_offset=3):

    d0s = []
    d1s = []
    gts = []
    p0_paths = []

    for data in tqdm(data_loader.load_data(), desc=name):
        if opt.use_gpu:
            data['ref'], data['p0'], data['p1'] = data_to_gpu(data)
        if test_type == 'xshifted':
            if shift_type == 'crop':
                data['p0'], data['ref'] = get_shifted_pair( data['p0'], data['ref'], n=n, shift_offset=shift_offset)
                data['p1'], data['ref'] = get_shifted_pair( data['p1'], data['ref'], n=n, shift_offset=shift_offset)
                d0s += func(data['ref'], data['p0']).data.cpu().numpy().flatten().tolist()
                d1s += func(data['ref'], data['p1']).data.cpu().numpy().flatten().tolist()
            gts += data['judge'].cpu().numpy().flatten().tolist()
        p0_paths += data['p0_path']

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)

    if (model_name == 'MSSSIM'):
        d0s_msssim = 1.-np.array(d0s)
        d1s_msssim = 1.-np.array(d1s)
        scores = (d0s_msssim<d1s_msssim)*(1.-gts) + (d1s_msssim<d0s_msssim)*gts + (d1s_msssim==d0s_msssim)*.5
    else:
        scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    print("model_name: ", model_name, save_dir)
    
    if test_type == 'xshifted':
        np.savetxt(save_dir + "/" + test_type
                + "_n-" + str(n) 
                + "_" + shift_direction 
                + "_" + shift_type
                + "_model-" + model_name
                + "_distortion-" + name.split("/")[-1]
                + "_results.csv", np.array([p0_paths, d0s, d1s, scores, gts], dtype='object').T, delimiter=",", fmt='%s')

    return(np.mean(scores), dict(d0s=d0s, d1s=d1s, gts=gts, scores=scores))

# initialize model
stlpips_metric = stlpips.LPIPS(net=opt.net, variant=opt.variant)
if(opt.use_gpu):
    print("gpu: "+str(opt.gpu_ids[0]))
    stlpips_metric.to(device=opt.gpu_ids[0])

# set model name
modelname = opt.net+'_'+opt.variant

# set results dir
save_dir = os.path.join('results', modelname)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# initialize data loader
for dataset in opt.datasets:
    data_loader = dl.CreateDataLoader(dataset, dataroot=gl_conf['2afc_rootdir'], dataset_mode=opt.dataset_mode, batch_size=opt.batch_size, load_size=opt.load_size, nThreads=opt.nThreads)
    
    # evaluate model on test type
    if(opt.test_type=='twoAFC'):
        (score, results_verbose) = score_2afc_dataset(data_loader=data_loader, 
                                                func=stlpips_metric.forward, 
                                                name=dataset, 
                                                model_name=modelname, 
                                                save_dir=save_dir)
    elif(opt.test_type=='xshifted'):
        (score, results_verbose) = score_shifted_2afc_dataset(data_loader=data_loader, 
                                                        func=stlpips_metric.forward, 
                                                        name=dataset, model_name=modelname, 
                                                        test_type=opt.test_type, 
                                                        shift_direction=opt.shift_direction, 
                                                        shift_type=opt.shift_type, 
                                                        n=opt.num_pixels_shifted, 
                                                        save_dir=save_dir, 
                                                        shift_offset=opt.shift_offset)

    # print results
    print('  Dataset [%s]: %.2f'%(dataset,100.*score))

