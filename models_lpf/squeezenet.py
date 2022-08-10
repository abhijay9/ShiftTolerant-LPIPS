import torch
import torch.nn as nn
import torch.nn.init as init
# from .utils import load_state_dict_from_url
from models_lpf import *

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, pad_fire=2):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, 
                                kernel_size=1, padding=pad_fire)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, padding=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=pad_fire)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))

        # print(x.size())
        # print(self.expand1x1_activation(self.expand1x1(x)).size())
        # print(self.expand1x1_activation(self.expand3x3(x)).size())

        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000, filter_size=3, pad_more=False, pad_n=1, pad_fire=2):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=pad_n),
                nn.ReLU(inplace=True),
                Downsample(filt_size=filter_size, stride=2, channels=96, pad_more=pad_more),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Downsample(filt_size=filter_size, stride=2, channels=96, pad_more=pad_more),
                Fire(96, 16, 64, 64, pad_fire),
                Fire(128, 16, 64, 64, pad_fire),
                Fire(128, 32, 128, 128, pad_fire),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Downsample(filt_size=filter_size, stride=2, channels=32, pad_more=pad_more),
                Fire(256, 32, 128, 128, pad_fire),
                Fire(256, 48, 192, 192, pad_fire),
                Fire(384, 48, 192, 192, pad_fire),
                Fire(384, 64, 256, 256, pad_fire),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=pad_more),
                Fire(512, 64, 256, 256, pad_fire),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=pad_n),
                nn.ReLU(inplace=True),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=pad_more),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=pad_more),
                Fire(64, 16, 64, 64, pad_fire),
                Fire(128, 16, 64, 64, pad_fire),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Downsample(filt_size=filter_size, stride=2, channels=128, pad_more=pad_more),
                Fire(128, 32, 128, 128, pad_fire),
                Fire(256, 32, 128, 128, pad_fire),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Downsample(filt_size=filter_size, stride=2, channels=256, pad_more=pad_more),
                Fire(256, 48, 192, 192, pad_fire),
                Fire(384, 48, 192, 192, pad_fire),
                Fire(384, 64, 256, 256, pad_fire),
                Fire(512, 64, 256, 256, pad_fire),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # if m.bias is not None:
                    #     nn.init.constant_(m.bias, 0)
                    if m is final_conv:
                        init.normal_(m.weight, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                else:
                    print('Not initializing')

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version, pretrained, progress, filter_size, pad_more, pad_n, pad_fire, **kwargs):
    model = SqueezeNet(version, 1000, filter_size, pad_more, pad_n, pad_fire, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, filter_size=3, pad_more=False, pad_n=1, pad_fire=2, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, filter_size, pad_more, pad_n, pad_fire, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, filter_size=3, pad_more=False, pad_n=1, pad_fire=2, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, filter_size, pad_more, pad_n, pad_fire, **kwargs)

