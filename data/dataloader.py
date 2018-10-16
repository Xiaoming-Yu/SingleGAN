import torch
import os
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def CreateDataLoader(opt):
    if opt.mode == 'base' or opt.mode == 'multimodal':
        sourceD, targetD = [0,1], [1,0]
    elif opt.mode == 'one2many':
        sourceD = [0 for i in range(opt.d_num-1)] + [i for i in range(1, opt.d_num)]
        targetD = [i for i in range(1, opt.d_num)] + [0 for i in range(opt.d_num-1)]
    elif opt.mode == 'many2many':
        sourceD = [i for i in range(opt.d_num) for j in range(opt.d_num)]
        targetD = [j for i in range(opt.d_num) for j in range(opt.d_num)]
    else:
        raise('mode:{} does not exist'.format(opt.mode))
    
    dataset = UpPairedDataset(opt.dataroot,
                            opt.loadSize,
                            opt.fineSize,
                            not opt.no_flip,
                            opt.isTrain,
                            sourceD=sourceD,
                            targetD=targetD
                            )
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batchSize,
                             shuffle=opt.isTrain,
                             drop_last=True,
                             num_workers=opt.nThreads)
                             
    return data_loader

class UpPairedDataset(Dataset):
    def __init__(self, image_path, loadSize, fineSize, isFlip, isTrain, sourceD=[0,1], targetD=[1,0]):
        self.image_path = image_path
        self.isTrain = isTrain
        self.fineSize = fineSize
        self.sourceD = sourceD
        self.targetD = targetD
        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        if isTrain:
            trs = [transforms.Resize(loadSize, interpolation=Image.ANTIALIAS), transforms.RandomCrop(fineSize)]
        else:
            trs = [transforms.Resize(loadSize, interpolation=Image.ANTIALIAS), transforms.CenterCrop(fineSize)]
        if isFlip:
            trs.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(trs)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.num_data = max(self.num)
        
    def preprocess(self):
        dirs = os.listdir(self.image_path)
        trainDirs = [dir for dir in dirs if 'train' in dir]
        testDirs = [dir for dir in dirs if 'test' in dir]
        assert len(trainDirs) >=  max(self.sourceD)+1 and len(trainDirs) >=  max(self.targetD)+1
        trainDirs.sort()
        testDirs.sort()
        self.filenames = []
        self.num = []
        if self.isTrain:
            for dir in trainDirs:
                filenames = glob("{}/{}/*.jpg".format(self.image_path,dir))
                random.shuffle(filenames)
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        else:
            for dir in testDirs:
                filenames = glob("{}/{}/*.jpg".format(self.image_path,dir))
                filenames.sort()
                self.filenames.append(filenames)
                self.num.append(len(filenames))
 
    def __getitem__(self, index):
        imgs = []
        for d in self.sourceD:
            index_d = index if index < self.num[d] else random.randint(0,self.num[d]-1)
            img = Image.open(self.filenames[d][index_d]).convert('RGB')
            img = self.transform(img)
            img = self.norm(img)
            imgs.append(img)
        return imgs, self.sourceD, self.targetD

    def __len__(self):
        return self.num_data

        
        
        
