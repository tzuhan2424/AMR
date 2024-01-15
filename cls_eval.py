import step.evaluateClassificationResult
import argparse

import importlib
import torch
import BUS.dataloader
from torch.utils.data import DataLoader
from misc import helper
from BUS.loadData import load
import numpy as np
from torchvision import transforms
from BUS import BC

if __name__ == '__main__':
    weight_name = 'sess/1006_01_res50_cam.pth'
    weight_name='/home/lintzuh@kean.edu/BUS/ReCAM/result_default5/0112_02_res50_cam'

    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_weights_name", default=weight_name, type=str)
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=32, type=int)


    args = parser.parse_args()




    #load the model
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()
    #prepare the sample

    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'



    train_images, train_labels, test_images, test_labels = load(dataRoot)  # this labels is for classfication

    
    val_dataset = BUS.dataloader.BUSClassificationDataset(test_images, np.array(test_labels),
                                                              resize_long=(320, 640), crop_size=512)
   
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False,num_workers=args.num_workers)

    # run the matrix
    helper.confusionMatrixOfClassify(val_data_loader, model, 2)

# if __name__ == '__main__':
#     weight_name = 'sess/0830_01_res50_cam.pth'

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cam_weights_name", default=weight_name, type=str)
#     parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
#     parser.add_argument("--cam_batch_size", default=16, type=int)

#     args = parser.parse_args()







#     #load the model
#     model = getattr(importlib.import_module(args.cam_network), 'Net')()
#     model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
#     model.eval()
#     #prepare the sample


#     train_images, train_labels, test_images, test_labels = load()  # this labels is for classfication


#     # create class
#     transform=transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         ])
#     val_dataset = BC.BC(test_images, test_labels, transform)

   
#     val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
#                                  shuffle=False)


#     # run the matrix
#     helper.confusionMatrixOfClassify(val_data_loader, model, 2)
