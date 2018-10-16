import sys
import os
from glob import glob
from PIL import Image
from tqdm import trange
from util import util
import shutil

image_path = sys.argv[1]
path_trainA = os.path.join(image_path,'trainA')
path_trainB = os.path.join(image_path,'trainB')
path_testA = os.path.join(image_path,'testA')
path_testB = os.path.join(image_path,'testB')
path = [path_trainA, path_trainB, path_testA, path_testB]
filenamesTrain = glob("{}/train/*.jpg".format(image_path))
filenamesTest = glob("{}/val/*.jpg".format(image_path))
filenamesTrain.sort()
filenamesTest.sort()
image = Image.open(filenamesTrain[0])
w,h = image.size
region_A = (0,0,w//2,h)
region_B = (w//2, 0, w, h)
files = [filenamesTrain, filenamesTest]
for i in trange(2):
    util.mkdirs(path[2*i])
    util.mkdirs(path[2*i+1])
    for j in trange(len(files[i])):
        img = Image.open(files[i][j])
        A = img.crop(region_A)
        A.save(os.path.join(path[2*i],'{}.jpg'.format(j)))
        B = img.crop(region_B)
        B.save(os.path.join(path[2*i+1],'{}.jpg'.format(j)))