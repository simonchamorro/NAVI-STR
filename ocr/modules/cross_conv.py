import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def crossconv2d(inputs, filters, stride=1, padding=1, dilation=1, groups=1):
    assert inputs.size(0) == filters.size(0), "oust!"
    _inputs  = inputs.unsqueeze(1)  # BxCixHxW --> Bx1xCixHxW
    _filters = filters              # BxCixCoxkHxkW
    tensors = [F.conv2d(_inputs[index], _filters[index], stride=stride, padding=padding, dilation=dilation, groups=groups) for index in range(inputs.size(0))]
    return torch.stack(tensors, dim=0).squeeze(1)

class CrossConvFilterGenerator(nn.Module):
    def __init__(self,
        inputs_dim=200,
        outputs_dim=((3*3*128*128) + (3*3*256*256) + (3*3*512*512) + (3*3*512*512))*2,  # ( (3x3x128x128) + (3x3x256x256) + (3x3x512x512) + (3x3x512x512) )x2
        latents_dim=(1024, 1024,),
        init_xavier=False):

        super().__init__()
        self.inputs_dim  = inputs_dim
        self.outputs_dim = outputs_dim  # BxCixCoxkHxkW
        self.latents_dim = latents_dim
        self.init_xavier = init_xavier

        self.built = False

    def build(self,):

        layers = [ nn.Linear(self.inputs_dim, self.latents_dim[0]), nn.ReLU() ]
        for i in range(0, len(self.latents_dim)-1):
            layers.append(nn.Linear(self.latents_dim[i], self.latents_dim[i+1]))
            layers.append(nn.Relu())
        layers.append( nn.Linear(self.latents_dim[-1], self.outputs_dim) )

        self.layers = nn.Sequential(*layers)

        if self.init_xavier:
            self.layers.apply(init_weights)

    def forward(self, inputs):
        if not self.built:
            self.build()
            self.built = True
        return self.layers(inputs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1_info = (3, 3, inplanes, planes)
        self.conv2_info = (3, 3, planes, planes)
        self.downsample = downsample
        self.stride = stride


        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.built

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def build(self,):
        # build the regular convolution layers
        self.conv1 = self._conv3x3(*self.conv1_info[-2:])
        self.conv2 = self._conv3x3(*self.conv2_info[-2:])

    def forward(self, x):            
        if len(x) == 2:
            residual, others = x
            x = x[0]

            shift1 = sum(self.conv1_info)
            shift2 = sum(self.conv2_info)

            gammas1 = others[:, :shift1]
            gammas2 = others[:, shift1:shift1+shift2]
            others  = others[:, shift1+shift2:]

            out = crossconv2d(inputs=out, filters=gammas1)
            out = self.bn1(out)
            out = self.relu(out)

            out = crossconv2d(inputs=out, filters=gammas2)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

            return out, others

        else:

            if not self.built:
                self.build()
                self.built = True
                
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