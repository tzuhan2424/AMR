import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio
import BUS.dataloader
# import voc12.dataloader
from misc import torchutils, indexing
from PIL import Image
from help.helper import getPathShortName


cudnn.enabled = True
palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]
def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0] #'/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT/benign/benign (270).png'
            orig_img_size = np.asarray(pack['size']) #[328, 500]
            edge, dp = model(pack['img'][0].cuda(non_blocking=True))



            shortname, type = getPathShortName(img_name)
            print('shortname', shortname)
            campath = os.path.join(args.cam_out_dir, type, shortname[:-4] + '.npy')
            cam_dict = np.load(campath, allow_pickle=True).item()

            cams = np.power(cam_dict['cam'], 1.5)
            
            keys = np.pad(cam_dict['keys'], (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()
            print('cam_downsized_values', cam_downsized_values.shape)
            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0, 0], :orig_img_size[1, 0]]

            # rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            out = Image.fromarray(rw_pred.astype(np.uint8), mode='P')
            out.putpalette(palette)

            subfolderPath = os.path.join(args.sem_seg_out_dir, type)
            os.makedirs(subfolderPath, exist_ok=True)
            seg_out_path = os.path.join(args.sem_seg_out_dir, type,  shortname[:-4])
            out.save(os.path.join(os.path.join(seg_out_path+'_palette.png')))
            imageio.imsave(os.path.join(seg_out_path+'.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()

    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()


    from Tzu_utilis.loaderHelper import readFromListWithoutNormal
    tarin_list_path='/home/lintzuh@kean.edu/BUS/ReCAM/record/train_images_list.txt'
    dataroot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    train_list, train_label= readFromListWithoutNormal(tarin_list_path, dataroot)



    n_gpus = torch.cuda.device_count()
    n_gpus=1
    dataset = BUS.dataloader.BUSClassificationDatasetMSF_resize(img_name_list_path = train_list,
                                                                crop_size=512,
                                                                rescale=(0.25),
                                                             cls_label = np.array(train_label),
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
