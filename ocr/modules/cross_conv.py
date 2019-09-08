import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def crossconv2d(inputs, filters, stride=1, padding=1, dilation=1, groups=1):
    assert inputs.size(0) == filters.size(0), "Mismatch of size along the batch axis between `inputs` and `filters`!"
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

        layers = [ nn.Linear(self.inputs_dim, self.latents_dim[0]), nn.ReLU() ]
        for i in range(0, len(self.latents_dim)-1):
            layers.append(nn.Linear(self.latents_dim[i], self.latents_dim[i+1]))
            layers.append(nn.Relu())
        layers.append( nn.Linear(self.latents_dim[-1], self.outputs_dim) )

        self.layers = nn.Sequential(*layers)

        if init_xavier:
            self.layers.apply(init_weights)

    def forward(self, inputs):
        return self.layers(inputs)
