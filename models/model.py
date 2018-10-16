import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

import functools
from .cbin import CBINorm2d
from .cbbn import CBBNorm2d


def get_norm_layer(layer_type='instance', num_con=2):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer    

def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun    
    
class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=0, bias=bias)
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))

class TrConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True, dilation=1, norm_layer=None, nl_layer=None):
        super(TrConv2dBlock, self).__init__()
        self.trConv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, dilation=dilation)
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.trConv(x)))

class Upsampling2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, type='Trp', norm_layer=None, nl_layer=None):
        super(Upsampling2dBlock, self).__init__()
        if type=='Trp':
            self.upsample = TrConv2dBlock(in_planes,out_planes,kernel_size=4,stride=2,padding=1,bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
        elif type=='Ner':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2dBlock(in_planes,out_planes,kernel_size=4, stride=1, padding=1, pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
                )
        else:
            raise('None Upsampling type {}'.format(type))
    def forward(self, x):
        return self.upsample(x)
    
def conv3x3(in_planes, out_planes, norm_layer=None, nl_layer=None):
    "3x3 convolution with padding"
    return Conv2dBlock(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)                     




################ Generator ###################       
class CResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, h_dim, c_norm_layer=None, nl_layer=None):
        super(CResidualBlock, self).__init__()
        self.c1 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(h_dim)
        self.l1 = nl_layer()
        self.c2 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim)

    def forward(self, input):
        x, c = input[0], input[1]
        y = self.l1(self.n1(self.c1(x),c))
        y = self.n2(self.c2(y),c)
        return [x + y,  c]

class SingleGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, nc=2, e_blocks=6, norm_type='instance', up_type='Trp'):
        super(SingleGenerator, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
        nl_layer = get_nl_layer(layer_type='relu')
        self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(ngf)
        self.a1 = nl_layer()
        self.c2 = Conv2dBlock(ngf, ngf*2, kernel_size=4, stride=2, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(ngf*2)
        self.a2 = nl_layer()
        self.c3 = Conv2dBlock(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, pad_type='reflect', bias=False)
        self.n3 = c_norm_layer(ngf*4)
        self.a3 = nl_layer()
        block = []
        for i in range(e_blocks):
            block.append(CResidualBlock(ngf*4, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.resBlocks =  nn.Sequential(*block)
        block = [Upsampling2dBlock(ngf*4,ngf*2,type=up_type,norm_layer=norm_layer,nl_layer=nl_layer)]
        block += [Upsampling2dBlock(ngf*2,ngf,type=up_type,norm_layer=norm_layer,nl_layer=nl_layer)]
        block +=[Conv2dBlock(ngf, output_nc, kernel_size=7, stride=1, padding=3, pad_type='reflect', bias=False,nl_layer=nn.Tanh)]
        self.upBlocks = nn.Sequential(*block)

    def forward(self, x, c):
        x = self.a1(self.n1(self.c1(x),c))
        x = self.a2(self.n2(self.c2(x),c))
        x = self.a3(self.n3(self.c3(x),c))
        x = self.resBlocks([x,c])[0]
        y = self.upBlocks(x)
        return y

################ Discriminator ##########################
class D_NET(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3,  norm_type='instance'):
        super(D_NET, self).__init__()
        norm_layer, _ = get_norm_layer(layer_type=norm_type)
        nl_layer = get_nl_layer('lrelu')
        block = [Conv2dBlock(input_nc, ndf, kernel_size=4,stride=2,padding=1,bias=False,nl_layer=nl_layer)]
        dim_in=ndf
        for n in range(1, block_num):
            dim_out = min(dim_in*2, ndf*8)
            block += [Conv2dBlock(dim_in, dim_out, kernel_size=4, stride=2, padding=1,bias=False,norm_layer=norm_layer,nl_layer=nl_layer)]
            dim_in = dim_out
        dim_out = min(dim_in*2, ndf*8)
        block += [Conv2dBlock(dim_in, 1, kernel_size=4, stride=1, padding=1,bias=True) ]
        self.conv = nn.Sequential(*block)
        
    def forward(self, x):
        return self.conv(x)
        

class D_NET_Multi(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, norm_type='instance'):
        super(D_NET_Multi, self).__init__()
        self.model_1 = D_NET(input_nc=input_nc, ndf=ndf, block_num=block_num, norm_type=norm_type)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model_2 = D_NET(input_nc=input_nc, ndf=ndf//2, block_num=block_num, norm_type=norm_type)
        
    def forward(self, x):
        pre1 = self.model_1(x)
        pre2 = self.model_2(self.down(x))
        return [pre1,pre2]
        
################ Encoder ##################################
def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)
    
def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, c_norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        self.cnorm1 = c_norm_layer(inplanes)
        self.nl1 = nl_layer()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.cnorm2 = c_norm_layer(inplanes)
        self.nl2 = nl_layer()
        self.cmp = convMeanpool(inplanes, outplanes)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, input):
        x, d = input
        out = self.cmp(self.nl2(self.cnorm2(self.conv1(self.nl1(self.cnorm1(x,d))),d)))
        out = out + self.shortcut(x)
        return [out,d]
        
class Encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, nef=64, nd=2, n_blocks=4, norm_type='instance'):
        # img 128*128 -> n_blocks=4 // img 256*256 -> n_blocks=5 
        super(Encoder, self).__init__()
        _, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nd)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='lrelu')
        self.entry = Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)
        conv_layers =[]
        for n in range(1, n_blocks):
            input_ndf = nef * min(max_ndf, n)  # 2**(n-1)
            output_ndf = nef * min(max_ndf, n+1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, c_norm_layer, nl_layer)]
        self.middle = nn.Sequential(*conv_layers)
        self.exit = nn.Sequential(*[nl_layer(), nn.AdaptiveAvgPool2d(1)])
        
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def forward(self, x, d):
        x_conv = self.exit(self.middle([self.entry(x),d])[0])
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        mu = self.fc(x_conv)
        logvar = self.fcVar(x_conv)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar