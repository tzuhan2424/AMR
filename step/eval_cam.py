
import numpy as np
import os
# from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from BUS.loadData import load
import imageio

from misc import imutils


def compute_iou(y_pred, y_true, num_classes):
    """Compute IoU for each class and return them in a list."""
    ious = []
    for cls in range(num_classes):
        pred_mask = np.where(y_pred == cls, 1, 0)
        true_mask = np.where(y_true == cls, 1, 0)

        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        if union == 0:  # Avoid division by zero
            iou = 0
        else:
            iou = intersection / union

        ious.append(iou)
    return ious


def run(args):
    # dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    train_images, train_labels, test_images, test_labels = load()  # this labels is for classfication

    preds = []
    labels = []
    n_images = 0
    for i, id in enumerate(train_images):
        if "normal" in id:
            continue
        n_images += 1
        cam_dict = np.load(args.cam_out_dir + id[23:-4] + '.npy', allow_pickle=True).item()
        cams = cam_dict['high_res']
        
        cams = np.expand_dims(cams, axis=0) if (cams.ndim < 3) else cams
        
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'], (1, 0), mode='constant')

        cls_labels = np.argmax(cams, axis=0)
        
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        label = np.int64(imageio.imread(id[:-4] + "_mask.png"))

        labels.append(label)
    iou = 0
    for i in range(len(preds)):
        iou += compute_iou(preds[i], labels[i], 2)[1]
    # confusion = calc_semantic_segmentation_confusion(preds, labels)
    #
    # gtj = confusion.sum(axis=1)
    # resj = confusion.sum(axis=0)
    # gtjresj = np.diag(confusion)
    # denominator = gtj + resj - gtjresj
    # iou = gtjresj / denominator


    print("threshold:", args.cam_eval_thres, 'miou:', iou/n_images, "i_imgs", n_images)
    # print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
