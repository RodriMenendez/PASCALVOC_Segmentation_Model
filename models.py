import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class Segmentation(nn.Module):
    def __init__(self, resnet, in_channels_list=[64, 128, 256, 512], out_channels=32):
        super().__init__()
        self.resnet = resnet
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.m = torchvision.ops.FeaturePyramidNetwork(in_channels_list, out_channels)

        # upsampling
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='nearest')
        self.upsample_32 = nn.Upsample(scale_factor=32, mode='nearest')

        # regularization
        self.dropout = nn.Dropout(0.2)

        # conv
        self.conv = nn.Conv2d(self.out_channels*4, 22, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(22)
        self.relu = nn.ReLU()


    def resnet_tail(self, input):
        """
        Goes through the first layers of the resnet model
        """
        return self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(input))))

    def FPN(self, input):
        input = self.resnet_tail(input)

        ly1_out = self.resnet.layer1(input)
        ly2_out = self.resnet.layer2(ly1_out)
        ly3_out = self.resnet.layer3(ly2_out)
        ly4_out = self.resnet.layer4(ly3_out)

        x = OrderedDict()
        x['layer1'] = ly1_out
        x['layer2'] = ly2_out
        x['layer3'] = ly3_out
        x['layer4'] = ly4_out

        return self.m(x)

    def logits(self, input):
        fpn_output = self.FPN(input)

        # make all outputs of fpn have the same height and width
        ly1_upsample = self.upsample_4(fpn_output['layer1'])
        ly2_upsample = self.upsample_8(fpn_output['layer2'])
        ly3_upsample = self.upsample_16(fpn_output['layer3'])
        ly4_upsample = self.upsample_32(fpn_output['layer4'])

        # concatenate into one tensor and use cnn to get correct number of channels
        output = torch.cat((ly1_upsample, ly2_upsample, ly3_upsample, ly4_upsample), 1)
        output = self.bn(self.conv(self.dropout(output)))

        return output

    def forward(self, input):
        logits = self.logits(input)
        return torch.argmax(logits, dim = 1)

def ResNet18():
    return torchvision.models.resnet18(weights='DEFAULT')