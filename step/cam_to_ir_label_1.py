
import os
import numpy as np
import imageio
import torch
from torch import multiprocessing
from torch.utils.data import DataLoader

import BUS.dataloader
from misc import torchutils, imutils
from PIL import Image
from BUS.loadData import load


palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

def _work(process_id, infer_dataset, args):
    from help.helper import getPathShortName


    visualize_intermediate_cam = False
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = pack['name'][0] #that is a full name
        img = pack['img'][0].numpy()

        #create subfolder
        shortname, type = getPathShortName(img_name)
        subfolder = os.path.join(args.ir_label_out_dir, type)
        os.makedirs(subfolder, exist_ok=True)

        
        campath = os.path.join(args.cam_out_dir, type, shortname[:-4] + '.npy')
        cam_dict = np.load(campath, allow_pickle=True).item()

        cams = cam_dict['high_res'] #cams.shape = [1, 598, 449]

        # a = torch.tensor([1])
        # keys = np.pad(a, (1, 0), mode='constant') #so after I modify keys = tensor([0,1])

        keys = np.pad(cam_dict['keys'], (1, 0), mode='constant') #cam_dict['keys'] = tensor([1]) for tumor, 0 for normal





        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres) #fg_conf_cam.shape = [2, 619, 763]
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0) #fg_conf_cam.shape = [619, 763]
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0]) #keys.shape[0] = 2, pred.shape = [619, 763]
        fg_conf = keys[pred]


        # bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
        # bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        # pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        # bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        # conf[fg_conf == 0] = 255
        # conf[bg_conf + fg_conf == 0] = 0

        
        # conf[fg_conf == 1] = 255
        # conf[bg_conf + fg_conf == 0] = 0

        conf_palette = fg_conf.copy()
        conf_palette[fg_conf == 1] = 1
        conf_palette[fg_conf == 0] = 0


        out = Image.fromarray(conf_palette.astype(np.uint8), mode='P')
        out.putpalette(palette)

        outpath = os.path.join(args.ir_label_out_dir, type, shortname[:-4])

        out.save(outpath + '_palette.png')
        # imageio.imwrite(outpath + '.png',
        #                 conf.astype(np.uint8))


        # if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
        #     print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):

    print('This is Tzu method!')

    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    train_images, train_labels, test_images, test_labels = load(dataRoot)  # this labels is for classfication
    
    def getFull(image_list, root='/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'):
        full_paths = []
        for image_name in image_list:
            if image_name[0] == 'b':
                type = 'benign'
            elif image_name[0] == 'm':
                type = 'malignant'
            else:
                type = 'normal'
            full_path = os.path.join(root, type,image_name)
            full_paths.append(full_path)
        return full_paths
    

    train_images_short=['benign (1).png', 'malignant (1).png']
    images = getFull(train_images_short)


    dataset = BUS.dataloader.BUSImageDataset(train_images, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')
