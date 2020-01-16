from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid
from models.model import D_NET_Multi, SingleGenerator, Encoder, weights_init
from util.loss import GANLoss, KL_loss
from util.util import tensor2im
import numpy as np

################## SingleGAN #############################
class SingleGAN():
    def name(self):
        return 'SingleGAN'

    def initialize(self, opt):
        torch.cuda.set_device(opt.gpu)
        cudnn.benchmark = True
        self.opt = opt
        self.build_models()
        
        
    def build_models(self):
        ################### generator #########################################
        self.G = SingleGenerator(input_nc=self.opt.input_nc, output_nc=self.opt.input_nc, ngf=self.opt.ngf, nc=self.opt.c_num+self.opt.d_num, e_blocks=self.opt.e_blocks, norm_type=self.opt.norm)
        ################### encoder ###########################################
        self.E =None
        if self.opt.mode == 'multimodal':
            self.E = Encoder(input_nc=self.opt.input_nc, output_nc=self.opt.c_num, nef=self.opt.nef, nd=self.opt.d_num, n_blocks=4, norm_type=self.opt.norm)
        if self.opt.isTrain:    
            ################### discriminators #####################################
            self.Ds = []
            for i in range(self.opt.d_num):
                self.Ds.append(D_NET_Multi(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=3,norm_type=self.opt.norm))
            ################### init_weights ########################################
            if self.opt.continue_train:
                self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                if self.E is not None:
                    self.E.load_state_dict(torch.load('{}/E_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                for i in range(self.opt.d_num):
                    self.Ds[i].load_state_dict(torch.load('{}/D_{}_{}.pth'.format(self.opt.model_dir, i, self.opt.which_epoch)))
            else:
                self.G.apply(weights_init(self.opt.init_type))
                if self.E is not None:
                    self.E.apply(weights_init(self.opt.init_type))
                for i in range(self.opt.d_num):
                    self.Ds[i].apply(weights_init(self.opt.init_type))
            ################### use GPU #############################################
            self.G.cuda()
            if self.E is not None:
                self.E.cuda()
            for i in range(self.opt.d_num):
                self.Ds[i].cuda()
            ################### set criterion ########################################
            self.criterionGAN = GANLoss(mse_loss=(self.opt.c_gan_mode == 'lsgan'))
            ################## define optimizers #####################################
            self.define_optimizers()
        else:
            self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            self.G.cuda()
            self.G.eval()
            if self.E is not None:
                self.E.load_state_dict(torch.load('{}/E_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.E.cuda()
                self.E.eval()
        
    def sample_latent_code(self, size):
        c = torch.cuda.FloatTensor(size).normal_()
        return Variable(c)
        
    def get_domain_code(self, domainLable):
        domainCode = torch.zeros([len(domainLable),self.opt.d_num])
        domainIndex_cache = [[] for i in range(self.opt.d_num)]
        for index in range(len(domainLable)):
            domainCode[index, domainLable[index]] = 1
            domainIndex_cache[domainLable[index]].append(index)
        domainIndex = []
        for index in domainIndex_cache:
            domainIndex.append(Variable(torch.LongTensor(index)).cuda())
        return Variable(domainCode).cuda(), domainIndex
        
    def define_optimizer(self, Net):
        return optim.Adam(Net.parameters(),
                                    lr=self.opt.lr,
                                    betas=(0.5, 0.999))
    def define_optimizers(self):
        self.G_opt = self.define_optimizer(self.G)
        self.E_opt = None
        if self.E is not None:
            self.E_opt = self.define_optimizer(self.E)
        self.Ds_opt = []
        for i in range(self.opt.d_num):
            self.Ds_opt.append(self.define_optimizer(self.Ds[i]))
    
    def update_lr(self, lr):
        for param_group in self.G_opt.param_groups:
            param_group['lr'] = lr
        if self.E_opt is not None:
            for param_group in self.E_opt.param_groups:
                param_group['lr'] = lr
        for i in range(self.opt.d_num):
            for param_group in self.Ds_opt[i].param_groups:
                param_group['lr'] = lr
                
    def save(self, name):
        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.opt.model_dir, name))
        if self.E_opt is not None:
            torch.save(self.E.state_dict(), '{}/E_{}.pth'.format(self.opt.model_dir, name))
        for i in range(self.opt.d_num):
            torch.save(self.Ds[i].state_dict(), '{}/D_{}_{}.pth'.format(self.opt.model_dir, i, name))
            
        
    def prepare_image(self, data):
        img, sourceD, targetD = data
        return Variable(torch.cat(img,0)).cuda(), torch.cat(sourceD,0), torch.cat(targetD,0)
    
    def translation(self, data):
        input, sourceD, targetD = self.prepare_image(data)
        sourceDC, sourceIndex = self.get_domain_code(sourceD)
        targetDC, targetIndex = self.get_domain_code(targetD)
        
        images, names =[], []
        for i in range(self.opt.d_num):
            images.append([tensor2im(input.index_select(0,sourceIndex[i])[0].data)])
            names.append(['D{}'.format(i)])
            
        if self.opt.mode == 'multimodal':
            for i in range(self.opt.n_samples):
                c_rand = self.sample_latent_code(torch.Size([input.size(0),self.opt.c_num]))
                targetC = torch.cat([targetDC, c_rand],1)
                output = self.G(input,targetC)
                for j in range(output.size(0)):
                    images[sourceD[j]].append(tensor2im(output[j].data))
                    names[sourceD[j]].append('{}to{}_{}'.format(sourceD[j],targetD[j],i)) 
        else:
            output = self.G(input,targetDC)
            for i in range(output.size(0)):
                images[sourceD[i]].append(tensor2im(output[i].data))
                names[sourceD[i]].append('{}to{}'.format(sourceD[i],targetD[i]))
            
        return  images, names
    
    def get_current_errors(self):
        dict = []
        for i in range(self.opt.d_num):
            dict += [('D_{}'.format(i), self.errDs[i].data.item())]
            dict += [('G_{}'.format(i), self.errGs[i].data.item())]
        dict += [('errCyc', self.errCyc.data.item())]
        if self.opt.lambda_ide > 0:
            dict += [('errIde', self.errIde.data.item())]
        if self.E is not None:
            dict += [('errKl', self.errKL.data.item())]
            dict += [('errCode', self.errCode.data.item())]
        return OrderedDict(dict)
        
    def get_current_visuals(self):
        real = make_grid(self.real.data,nrow=self.real.size(0),padding=0)
        fake = make_grid(self.fake.data,nrow=self.real.size(0),padding=0)
        cyc = make_grid(self.cyc.data,nrow=self.real.size(0),padding=0)
        img = [real,fake,cyc]
        name = 'rsal,fake,cyc'
        if self.opt.lambda_ide > 0:
            ide = make_grid(self.ide.data,nrow=self.real.size(0),padding=0)
            img.append(ide)
            name +=',ide'
        img = torch.cat(img,1)
        return OrderedDict([(name,tensor2im(img))])
        
    def update_D(self, D, D_opt, real, fake):
        D.zero_grad()
        pred_fake = D(fake.detach())
        pred_real = D(real)
        errD = self.criterionGAN(pred_fake,False) + self.criterionGAN(pred_real,True)
        errD.backward()
        D_opt.step()
        return errD
        
    def calculate_G(self, D, fake):
        pred_fake = D(fake)
        errG = self.criterionGAN(pred_fake,True)
        return errG
        
    def update_model(self,data):
        ### prepare data ###
        self.real, sourceD, targetD = self.prepare_image(data)
        sourceDC, self.sourceIndex = self.get_domain_code(sourceD)
        targetDC, self.targetIndex = self.get_domain_code(targetD)
        sourceC, targetC = sourceDC, targetDC
        ### generate image ###
        if self.E is not None:
            c_enc, mu, logvar = self.E(self.real,sourceDC)
            c_rand = self.sample_latent_code(c_enc.size())
            sourceC = torch.cat([sourceDC, c_enc],1)
            targetC = torch.cat([targetDC, c_rand],1)
        self.fake = self.G(self.real,targetC)
        self.cyc = self.G(self.fake,sourceC)
        if self.E is not None:
            _, mu_enc, _ = self.E(self.fake,targetDC)
        if self.opt.lambda_ide > 0:
            self.ide = self.G(self.real,sourceC)
        ### update D ###
        self.errDs = []
        for i in range(self.opt.d_num):
            errD = self.update_D(self.Ds[i], self.Ds_opt[i], self.real.index_select(0,self.sourceIndex[i]), self.fake.index_select(0,self.targetIndex[i]))
            self.errDs.append(errD)
        ### update G ###
        self.errGs, self.errKl, self.errCode, errG_total = [], 0, 0, 0
        self.G.zero_grad()
        for i in range(self.opt.d_num):
            errG = self.calculate_G(self.Ds[i], self.fake.index_select(0,self.targetIndex[i]))
            errG_total += errG
            self.errGs.append(errG)
        self.errCyc = torch.mean(torch.abs(self.cyc-self.real)) *  self.opt.lambda_cyc
        errG_total += self.errCyc
        if self.opt.lambda_ide > 0:
            self.errIde = torch.mean(torch.abs(self.ide-self.real)) *  self.opt.lambda_ide
            errG_total += self.errIde
        if self.E is not None:
            self.E.zero_grad()
            self.errKL = KL_loss(mu,logvar) * self.opt.lambda_kl
            errG_total += self.errKL
            errG_total.backward(retain_graph=True)
            self.G_opt.step()
            self.E_opt.step()
            self.G.zero_grad()
            self.E.zero_grad()
            self.errCode = torch.mean(torch.abs(mu_enc - c_rand)) * self.opt.lambda_c
            self.errCode.backward()
            self.G_opt.step()
        else:
            errG_total.backward()
            self.G_opt.step()
