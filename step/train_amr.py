import os
import cv2
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import importlib
import numpy as np
from BUS.loadData import load

import BUS.dataloader
from misc import pyutils, torchutils

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img'].cuda()

            label = pack['label'].cuda(non_blocking=True)
            one_hot_labels = F.one_hot(label, num_classes=2)

            logits1, cam1, logits2, cam2 = model(img)
            losscls1 = F.multilabel_soft_margin_loss(logits1, one_hot_labels)
            losscls2 = F.multilabel_soft_margin_loss(logits2, one_hot_labels)

            loss_cps = torch.mean(torch.abs(cam1[1:,:,:]-cam2[1:,:,:]))

            loss1 = 0.5* losscls1 + 0.5* losscls2 + 0.05* loss_cps

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):
    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'

    train_images, train_labels, test_images, test_labels = load(dataRoot)  # this labels is for classfication

    model = getattr(importlib.import_module(args.amr_network), 'Net')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=False)
    

    train_dataset = BUS.dataloader.BUSClassificationDataset(train_images, np.array(train_labels),
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = BUS.dataloader.BUSClassificationDataset(test_images, np.array(test_labels),
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
 
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[2], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
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
            one_hot_labels = F.one_hot(label, num_classes=2)

            model.zero_grad()
            logits1, cam1, logits2, cam2 = model(img)

            optimizer.zero_grad()

            losscls1 = F.multilabel_soft_margin_loss(logits1, one_hot_labels)
            losscls2 = F.multilabel_soft_margin_loss(logits2, one_hot_labels)
            
            loss_cps = torch.mean(torch.abs(cam1 - cam2))

            loss = 0.5 * losscls1 + 0.5 * losscls2 + 0.05 * loss_cps

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

    torch.save(model.state_dict(), args.amr_weights_name + '.pth')
    torch.cuda.empty_cache()