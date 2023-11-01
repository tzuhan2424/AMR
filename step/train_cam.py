import cv2
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import importlib
from BUS.loadData import load
import BUS.dataloader
from misc import pyutils, torchutils
from torch import autograd
import os



def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img'].cuda()

            label = pack['label'].cuda(non_blocking=True)
            one_hot_labels = F.one_hot(label, num_classes=2)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, one_hot_labels)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):
    from torchvision import transforms
    from BUS import BC


    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    
    # provide root:
    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    train_images, train_labels, test_images, test_labels = load(dataRoot)  # this labels is for classfication
    #[/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT/benign/benign (1).png]

    
    train_dataset = BUS.dataloader.BUSClassificationDataset(train_images, np.array(train_labels),
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random", cutout=128)
 
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = BUS.dataloader.BUSClassificationDataset(test_images, np.array(test_labels),resize_long=(320, 640),
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)


    model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            model.zero_grad()
            x = model(img)
            one_hot_labels = F.one_hot(label, num_classes=2)
            optimizer.zero_grad()

            loss = F.multilabel_soft_margin_loss(x, one_hot_labels)

            loss.backward()
            avg_meter.add({'loss1': loss.item()})


            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()