from collections import namedtuple
from torchvision import models as tv
from IPython import embed
import torch.nn as nn
from models_lpf import *
# torch.autograd.set_detect_anomaly(True)

class alexnet(nn.Module):
    def __init__(self, variant='vanilla', filter_size=3):
        super(alexnet, self).__init__()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        if variant == 'vanilla':
            features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True), #1
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True), #4
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #7
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #9
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #11
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
            for x in range(2):
                self.slice1.add_module(str(x), features[x])
            for x in range(2, 5):
                self.slice2.add_module(str(x), features[x])
            for x in range(5, 8):
                self.slice3.add_module(str(x), features[x])
            for x in range(8, 10):
                self.slice4.add_module(str(x), features[x])
            for x in range(10, 12):
                self.slice5.add_module(str(x), features[x])

        elif variant == 'antialiased':
            features= nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                nn.ReLU(inplace=True), #1
                Downsample(filt_size=filter_size, stride=2, channels=64),
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=64),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True), #6
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=192),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #10
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #12
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #14
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=256)
            )
            for x in range(2):
                self.slice1.add_module(str(x), features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), features[x])
            for x in range(11, 13):
                self.slice4.add_module(str(x), features[x])
            for x in range(13, 15):
                self.slice5.add_module(str(x), features[x])
        
        elif variant == 'shift_tolerant': # antialiased_blurpoolReflectionPad2_conv1stride1_blurAfter
            features= nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=2),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=True),
                nn.ReLU(inplace=True), #2
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=True),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True), #6
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=192, pad_more=True),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #10
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #12
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), #14
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=256, pad_more=True)
            )
            for x in range(3):
                self.slice1.add_module(str(x), features[x])
            for x in range(3, 7):
                self.slice2.add_module(str(x), features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), features[x])
            for x in range(11, 13):
                self.slice4.add_module(str(x), features[x])
            for x in range(13, 15):
                self.slice5.add_module(str(x), features[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out

class vggnet(nn.Module):
    def __init__(self, variant='vanilla', filter_size=3, requires_grad = True):
        super(vggnet, self).__init__()
        
        filter_size = 3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        if variant == 'vanilla':
            vgg_features = tv.vgg16(pretrained=False).features
            for x in range(4):
                self.slice1.add_module(str(x), vgg_features[x])
            for x in range(4, 9):
                self.slice2.add_module(str(x), vgg_features[x])
            for x in range(9, 16):
                self.slice3.add_module(str(x), vgg_features[x])
            for x in range(16, 23):
                self.slice4.add_module(str(x), vgg_features[x])
            for x in range(23, 30):
                self.slice5.add_module(str(x), vgg_features[x])
            if not requires_grad:
                for param in self.parameters():
                    param.requires_grad = False
                    
        elif variant == 'shift_tolerant':
            vgg_features = vgg16(filter_size=filter_size, pad_more=True).features
            for x in range(4):
                self.slice1.add_module(str(x), vgg_features[x])
            for x in range(4, 10):
                self.slice2.add_module(str(x), vgg_features[x])
            for x in range(10, 18):
                self.slice3.add_module(str(x), vgg_features[x])
            for x in range(18, 26):
                self.slice4.add_module(str(x), vgg_features[x])
            for x in range(26, 34):
                self.slice5.add_module(str(x), vgg_features[x])
            if not requires_grad:
                for param in self.parameters():
                    param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

class squeezenet11(nn.Module):
    def __init__(self, variant='vanilla', requires_grad = True):
        super(squeezenet11, self).__init__()
        
        filter_size = 3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()
        self.slice7 = nn.Sequential()
        self.N_slices = 7

        if variant == 'vanilla':
            squeezenet_features = tv.squeezenet1_1(pretrained=False).features
            for x in range(2):
                self.slice1.add_module(str(x), squeezenet_features[x])
            for x in range(2, 5):
                self.slice2.add_module(str(x), squeezenet_features[x])
            for x in range(5, 8):
                self.slice3.add_module(str(x), squeezenet_features[x])
            for x in range(8, 10):
                self.slice4.add_module(str(x), squeezenet_features[x])
            for x in range(10, 11):
                self.slice5.add_module(str(x), squeezenet_features[x])
            for x in range(11, 12):
                self.slice6.add_module(str(x), squeezenet_features[x])
            for x in range(12, 13):
                self.slice7.add_module(str(x), squeezenet_features[x])
            if not requires_grad:
                for param in self.parameters():
                    param.requires_grad = False
                    
        elif variant == 'antialiased':
            squeezenet_features = squeezenet1_1(pretrained=False, filter_size=filter_size, pad_more=False, pad_n=1, pad_fire=2).features
            for x in range(2):
                self.slice1.add_module(str(x), squeezenet_features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), squeezenet_features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), squeezenet_features[x])
            for x in range(11, 14):
                self.slice4.add_module(str(x), squeezenet_features[x])
            for x in range(14, 15):
                self.slice5.add_module(str(x), squeezenet_features[x])
            for x in range(15, 16):
                self.slice6.add_module(str(x), squeezenet_features[x])
            for x in range(16, 17):
                self.slice7.add_module(str(x), squeezenet_features[x])
            if not requires_grad:
                for param in self.parameters():
                    param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out

class resnet18net(nn.Module):
    def __init__(self, variant='vanilla'):
        super(resnet18net, self).__init__()

        filter_size = 3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        if variant == 'vanilla':
            self.net = tv.resnet18(pretrained=False)

            self.conv1 = self.net.conv1
            self.bn1 = self.net.bn1
            self.relu = self.net.relu
            self.maxpool = self.net.maxpool
            self.layer1 = self.net.layer1
            self.layer2 = self.net.layer2
            self.layer3 = self.net.layer3
            self.layer4 = self.net.layer4
            
            resnet_features = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.maxpool,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4
            )
            for x in range(3):
                self.slice1.add_module(str(x), resnet_features[x])
            for x in range(3, 5):
                self.slice2.add_module(str(x), resnet_features[x])
            for x in range(5, 6):
                self.slice3.add_module(str(x), resnet_features[x])
            for x in range(6, 7):
                self.slice4.add_module(str(x), resnet_features[x])
            for x in range(7, 8):
                self.slice5.add_module(str(x), resnet_features[x])
                
        elif variant == 'antialiased':
            self.net = resnet18(filter_size=filter_size, pool_only=False)

            self.conv1 = self.net.conv1
            self.bn1 = self.net.bn1
            self.relu = self.net.relu
            self.maxpool = self.net.maxpool
            self.layer1 = self.net.layer1
            self.layer2 = self.net.layer2
            self.layer3 = self.net.layer3
            self.layer4 = self.net.layer4

            resnet_features = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.maxpool,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4
            )
            for x in range(3):
                self.slice1.add_module(str(x), resnet_features[x])
            for x in range(3, 5):
                self.slice2.add_module(str(x), resnet_features[x])
            for x in range(5, 6):
                self.slice3.add_module(str(x), resnet_features[x])
            for x in range(6, 7):
                self.slice4.add_module(str(x), resnet_features[x])
            for x in range(7, 8):
                self.slice5.add_module(str(x), resnet_features[x])

    def forward(self, X):
        
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_conv2 = h
        h = self.slice3(h)
        h_conv3 = h
        h = self.slice4(h)
        h_conv4 = h
        h = self.slice5(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out
    
    
