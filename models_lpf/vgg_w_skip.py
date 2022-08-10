import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models_lpf import *

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
    #                 bias=False)
    if stride==2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1,
                    bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                    bias=False)

class BasicBlock(nn.Module):
    
    def __init__(self, conv3x3_layers, inchannels_block, outchannels_block, pool_type, w_skip, skip_pool_type, stride=1, filter_size=3, pad_more=True):
        super(BasicBlock, self).__init__()
        
        self.w_skip = w_skip
        
        if pool_type == "blurpool":
            conv3x3_layers += [Downsample(filt_size=filter_size, stride=2, channels=outchannels_block, pad_more=pad_more),]
        elif pool_type == "max_w_blurpool":
            conv3x3_layers += [nn.MaxPool2d(kernel_size=2, stride=1), 
                Downsample(filt_size=filter_size, stride=2, channels=outchannels_block, pad_more=pad_more),]

        self.conv3x3_layers = nn.Sequential(*conv3x3_layers)
        
        if self.w_skip:
                       
            self.skip_layers = [conv1x1(inchannels_block, outchannels_block, stride)]
            # if skip_pool_type != pool_type:
            #     pad_more = False
            if skip_pool_type == "blurpool":
                self.skip_layers += [Downsample(filt_size=filter_size, stride=2, channels=outchannels_block, pad_more=pad_more),]
            elif skip_pool_type == "max_w_blurpool":
                self.skip_layers += [nn.MaxPool2d(kernel_size=2, stride=1), 
                    Downsample(filt_size=filter_size, stride=2, channels=outchannels_block, pad_more=pad_more),]
            
            self.skip_layers = nn.Sequential(*self.skip_layers)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv3x3_layers(x)
        if self.w_skip:
            out_skip = self.skip_layers(x)
            out += out_skip
        out = self.relu(out)
        return out

class VGG_W_SKIP(nn.Module):

    def __init__(self, block=BasicBlock, vggcfg=cfg['D'], filter_size=3, pad_more=False, fconv=False, pool_type="blurpool", skip_pool_type="blurpool", w_skip=True, stride=1, init_weights=True):
        super(VGG_W_SKIP, self).__init__()
        
        i = 0
        self.layer = []
        print(self.layer)
        layers = []
        inchannels = 3
        inchannels_block = inchannels
        for k, v in enumerate(vggcfg):
            if v == 'M':
                self.layer.append(block(layers, inchannels_block, outchannels_block, pool_type, w_skip, skip_pool_type, stride=stride, filter_size=3, pad_more=pad_more))
                layers = []
                inchannels_block = inchannels
            else:
                conv2d = conv3x3(inchannels, v)
                if vggcfg[k+1] != 'M':
                    layers += [conv2d, nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d]

                inchannels = v
                outchannels_block = v
                
        self.layer = nn.Sequential(*(self.layer))
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    # def forward(self, x):
    #     x = self.layers[0](x)
    #     x = self.layers[1](x)
    #     x = self.layers[2](x)
    #     x = self.layers[3](x)
    #     x = self.layers[4](x)
    #     return x

def vgg16_w_skip(pretrained=False, filter_size=3, pad_more=True, pool_type="blurpool", skip_pool_type="blurpool", w_skip=True, init_weights=True, stride=1, **kwargs):
    
    # if pretrained:
    #     kwargs['init_weights'] = False
    
    model = VGG_W_SKIP(block=BasicBlock, vggcfg=cfg['D'], filter_size=filter_size, pad_more=pad_more, pool_type=pool_type, skip_pool_type=skip_pool_type, w_skip=w_skip, stride=stride, init_weights=True, **kwargs)
    
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    
    return model