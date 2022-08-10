
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from . import pretrained_networks as pn
import torch.nn
import stlpips
from stlpips import invariant_networks as inv_net
from torchsummary import summary
import audtorch
import sys

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', variant='', version='0.1', lpips=True, spatial=False,
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True, blur_filter_size=3):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if variant == '':
            if(self.pnet_type in ['vgg','vgg16']):
                net_type = pn.vgg16
                self.chns = [64,128,256,512,512]
            elif(self.pnet_type=='alex'):
                net_type = pn.alexnet
                self.chns = [64,192,384,256,256]
            elif(self.pnet_type=='squeeze'):
                net_type = pn.squeezenet
                self.chns = [64,128,256,384,384,512,512]
            
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
            
        else:
            if(self.pnet_type in ['vgg','vgg16']):
                net_type = inv_net.vggnet
                self.chns = [64,128,256,512,512]
            elif(self.pnet_type=='alex'):
                net_type = inv_net.alexnet
                self.chns = [64,192,384,256,256]
            elif(self.pnet_type=='squeeze'):
                net_type = inv_net.squeezenet
                self.chns = [64,128,256,384,384,512,512]
            elif(self.pnet_type == 'resnet'):
                net_type = inv_net.resnet18
                self.chns = [64,64,128,256,512]
            
            self.net = net_type(variant=variant, filter_size=blur_filter_size)
            # pretrained = False # when training
            
        self.L = len(self.chns)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    if variant == '':
                        model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))
                    else:
                        version = "ST0.0"
                        model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s_%s.pth'%(version,net,variant)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = stlpips.normalize_tensor(outs0[kk]), stlpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

class DmosMse(nn.Module):
    def __init__(self):
        super(DmosMse, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, dDmos, judgeMos):
        self.logit = dDmos
        return self.loss(self.logit, judgeMos)

class DmosSrcc(nn.Module):
    def __init__(self):
        super(DmosSrcc, self).__init__()
        self.loss = audtorch.metrics.functional.pearsonr

    def forward(self, dDmos, judgeMos):
        self.logit = dDmos
        return self.loss(self.logit.flatten(), judgeMos.flatten(), batch_first=True)

class RankingLoss(nn.Module):
    def __init__(self, chn_mid=32, loss_func=''):
        super(RankingLoss, self).__init__()

        print("loss_func: ", loss_func)
        if loss_func == 'BCE':
            self.loss = torch.nn.BCELoss()
        elif loss_func == 'MSE':
            self.loss = torch.nn.MSELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = d0/(d0+d1)
        return self.loss(self.logit, per)

# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace

class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = stlpips.l2(stlpips.tensor2np(stlpips.tensor2tensorlab(in0.data,to_norm=False)), 
                stlpips.tensor2np(stlpips.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = stlpips.dssim(1.*stlpips.tensor2im(in0.data), 1.*stlpips.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = stlpips.dssim(stlpips.tensor2np(stlpips.tensor2tensorlab(in0.data,to_norm=False)), 
                stlpips.tensor2np(stlpips.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
    # Model summary
    print(summary(net,[(3, 64, 64),(3, 64, 64)]))
    # print(summary(net,[(3, 64, 61),(3, 64, 61)]))
    # print(summary(net,[(3, 256, 253),(3, 256, 253)]))
    # print(summary(net,[(3, 256, 256),(3, 256, 256)]))
    # print(summary(net,[(3, 256, 255),(3, 256, 255)]))
    # print(summary(net,[(3, 256, 254),(3, 256, 254)]))
    # print(summary(net,[(3, 256, 253),(3, 256, 253)]))
    # print(summary(net,[(3, 256, 252),(3, 256, 252)]))