import os
from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from util.visualizer import save_images
from itertools import islice
from models.single_gan import SingleGAN
from util import html, util

    
opt = TestOptions().parse()
opt.no_flip = True  
opt.batchSize = 1

data_loader = CreateDataLoader(opt)

model = SingleGAN()
model.initialize(opt)

web_dir = os.path.join(opt.results_dir, 'test')
webpage = html.HTML(web_dir, 'task {}'.format(opt.name))

for i, data in enumerate(islice(data_loader, opt.how_many)):
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))
    all_images, all_names = model.translation(data)
    img_path = 'image%3.3i' % i
    save_images(webpage, all_images, all_names, img_path, None, width=opt.fineSize)

        
webpage.save()                