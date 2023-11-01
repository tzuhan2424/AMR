import importlib
import torch
import BUS.dataloader
from torch.utils.data import DataLoader
from misc import helper
from BUS.loadData import load
import numpy as np
from torchvision import transforms
from BUS import BC
def run(args):

    #load the model
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()
    #prepare the sample


    train_images, train_labels, test_images, test_labels = load()  # this labels is for classfication


    # create class
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    val_dataset = BC.BC(test_images, test_labels, transform)

    # val_dataset = BUS.dataloader.BUSClassificationDataset(test_images, np.array(test_labels),
    #                                                           resize_long=(610, 620), crop_size=512)
    # val_dataset = BUS.dataloader.BUSClassificationDataset(test_images, np.array(test_labels),crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
    #                              shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)



    # run the matrix
    helper.confusionMatrixOfClassify(val_data_loader, model, 2)