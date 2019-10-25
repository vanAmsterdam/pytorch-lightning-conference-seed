'''
Define different PyTorch modules for the different targets, including loss functions
'''

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Poisson
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
import math

# some basic modules
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Identity(nn.Module):
    def forward(self, x):
        return x

class RGBSlice(nn.Module):
    '''
    take an (1*x*y*z) shaped scan and turn it into (3,z/3,x,y) shaped 'rgb-movie'
    '''
    def forward(self, x):
        batch_size = x.shape[0]
        x_size     = x.shape[2]
        y_size     = x.shape[3]
        z_size     = x.shape[4]
        try:
            assert z_size % 3 == 0 
        except:
            ValueError(f"should always input tensors that have a z-dim such that z-dim mod 3 == 0, found {x.shape}")

        n_slices = int(z_size / 3)

        # squeeze of the 1 channel dimension
        x = x.squeeze(1)

        # premute channels (0=batch, 1=x, 2=y, 3=z) to (batch, z, x, y)
        x = x.permute(0, 3, 1, 2)

        # squeeze the z slices in to packs of 3
        x = x.view(batch_size, n_slices, 3, x_size, y_size)

        # permute again (0=batch, 1=z/3, 2=rgb-stack, 3=x, 4=y) to (batch,3,z/3,x,y) for the required output
        return x.permute(0,2,1,3,4)

# add 3D ResNet definition
# ported from https://github.com/kenshohara/3D-ResNets-PyTorch

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = nn.Parameter(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''
    basic definition of 3D ResNet Module
    encoder: skip fcs
    '''
    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 encode_only=False,
                 num_classes=400,
                 pretrained=False):
        
        self.inplanes    = 64
        self.encode_only = encode_only
        self.pretrained  = pretrained
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        if (not encode_only) or pretrained:
            self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.rgbslicer = RGBSlice()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.pretrained:
            x = self.rgbslicer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if self.encode_only:
            return x

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(pretrainedpath=None, **kwargs):
    """Constructs a ResNet-18 model.
    """
    if pretrainedpath:
        pretrained=True
    model = ResNet(BasicBlock, [2, 2, 2, 2], shortcut_type='A', pretrained=pretrained, **kwargs)
    if pretrainedpath is not None:
        print(f"loading state dict from {pretrainedpath}")
        checkpoint = torch.load(pretrainedpath)
        model.load_state_dict(rename_state_dict_weights(checkpoint['state_dict']))
    return model



def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

# add definition of densenet

def densenet121(pretrainedpath=None, **kwargs):
    if pretrainedpath is not None:
        model = DenseNet(
            num_init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_classes=400,
            **kwargs)
        print(f"loading state dict from {pretrainedpath}")
        checkpoint = torch.load(pretrainedpath)
        model.load_state_dict(rename_state_dict_weights(checkpoint['state_dict']))
    else:
        model = DenseNet(
            num_init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs)
    return model


def densenet264(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        **kwargs)
    return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        # super(_DenseLayer, self).__init__()
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 encode_only=False
                 ):
        self.encode_only=True
        super(DenseNet, self).__init__()

        # RBG slice 
        self.rgbslicer = RGBSlice()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     3,
                     num_init_features,
                     kernel_size=7,
                     stride=(1, 2, 2),
                     padding=(3, 3, 3),
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.rgbslicer(x)
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)

        if self.encode_only:
            return x

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

# create helper to rename some layers from the pre-trained weights
def rename_state_dict_weights(state_dict):
    return {fix_layer_name(k[7:]): v for k, v in state_dict.items()}

def fix_layer_name(x: str) -> str:
    # function that takes in a string and fixes it when needed
    # these were incompatible with current pytorch version:
    # conv.1, conv.2, relu.1, relu.2, norm.1, norm.2
    bad_names  = ['conv.1', 'conv.2', 'relu.1', 'relu.2', 'norm.1', 'norm.2']
    good_names = ['conv1', 'conv2', 'relu1', 'relu2', 'norm1', 'norm2']
    x_out = x
    for bad_name, good_name in zip(bad_names, good_names):
        x_split = x.split(bad_name)
        if len(x_split) == 1:
            pass
        elif len(x_split) == 2:
            x_out = x_split[0] + good_name + x_split[1]
        else:
            raise ValueError(f'this function is not supposed to work when multiple occurences of bad name: {bad_name} occur in x: {x}')
        if x_out != x:
            return x_out
    return x_out

# metrics
def auc(outputs, labels):
    """
    outputs, labels: np.ndarray
    area under ROC curve
    code from tnt/meter/aucmeter
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # case when number of elements added are 0
    if outputs.shape[0] == 0:
        return (0.5, 0.0, 0.0)
    
    if outputs.ndim > 1:
        if outputs.shape[1] == 2:
            outputs = outputs[:,1]
        else:
            outputs = np.squeeze(outputs)

    # sorting the arrays
    scores, sortind = torch.sort(torch.from_numpy(
        outputs), dim=0, descending=True)
    scores = scores.numpy()
    sortind = sortind.numpy()

    # creating the roc curve
    tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
    fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

    for i in range(1, scores.size + 1):
        if labels[sortind[i - 1]] == 1:
            tpr[i] = tpr[i - 1] + 1
            fpr[i] = fpr[i - 1]
        else:
            tpr[i] = tpr[i - 1]
            fpr[i] = fpr[i - 1] + 1

    tpr /= (labels.sum() * 1.0)
    fpr /= ((labels - 1.0).sum() * -1.0)

    # calculating area under curve using trapezoidal rule
    n = tpr.shape[0]
    h = fpr[1:n] - fpr[0:n - 1]
    sum_h = np.zeros(fpr.shape)
    sum_h[0:n - 1] = h
    sum_h[1:n] += h
    area = (sum_h * tpr).sum() / 2.0

    # return (area, tpr, fpr)
    return area

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if outputs.ndim > 1 and outputs.shape[outputs.ndim-1] > 1:
        outputs = np.argmax(outputs, axis=1)
    else:
        outputs = (outputs > 0).astype(np.int16).squeeze()
    
    return np.mean(outputs==labels.astype(np.int16))

