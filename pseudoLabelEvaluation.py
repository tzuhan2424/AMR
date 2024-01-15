#%%
"""
Tzu-Han Lin
evaluation of pseudo label and gt
"""
from PIL import Image
import numpy as np
def metrics(predict, label):
    import numpy as np
    #tumor
    predict = predict.astype("bool")
    label = label.astype("bool")
    intersection = predict * label
    sum_inter = np.sum(intersection)
    union = predict + label
    sum_union = np.sum(union)
    IoU = sum_inter / sum_union
    
    #background
    bg_predict = ~predict
    bg_label = ~label
    bg_intersection = bg_predict * bg_label
    bg_sum_inter = np.sum(bg_intersection)
    bg_union = bg_predict + bg_label
    bg_sum_union = np.sum(bg_union)
    bg_IOU = bg_sum_inter / bg_sum_union



    return IoU, bg_IOU

def iou_labelcomparsion_p(label_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f04'
                        ,dataroot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
                        ,train_list_path='/home/lintzuh@kean.edu/BUS/ReCAM/record/train_images_list.txt'):
    from Tzu_utilis.loaderHelper import readFromListWithoutNormalAndShortName
    from help.helper import getFullName,getPseudoLabelPathFromShortName_palette

    train_list, train_label= readFromListWithoutNormalAndShortName(train_list_path, dataroot) #train_list has short name: benign (1).png
    
    
    tumor=[]
    bgList=[]

    
    for shortname in train_list:
        gt = getFullName(shortname[:-4], dataroot)[:-4]+'_mask.png'
        pseudoLabel = getPseudoLabelPathFromShortName_palette(shortname[:-4], label_path) 

        gt_image = np.array(Image.open(gt).split()[0])
        pseudoLabel_image = np.array(Image.open(pseudoLabel).split()[0])
        tumor_iou, bg_iou = metrics(pseudoLabel_image, gt_image)
        tumor.append(tumor_iou)
        bgList.append(bg_iou)

    average_bg = sum(bgList) / len(bgList)
    average_tm = sum(tumor) / len(tumor)
    print('average of background iou', average_bg)
    print('average of tumor iou', average_tm)

    print('miou:', (average_tm+average_bg)/2)

def iou_labelcomparsion(label_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f04'
                        ,dataroot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
                        ,train_list_path='/home/lintzuh@kean.edu/BUS/ReCAM/record/train_images_list.txt'):
    from Tzu_utilis.loaderHelper import readFromListWithoutNormalAndShortName
    from help.helper import getFullName,getPseudoLabelPathFromShortName

    train_list, train_label= readFromListWithoutNormalAndShortName(train_list_path, dataroot) #train_list has short name: benign (1).png
    
    
    tumor=[]
    bgList=[]

    
    for shortname in train_list:
        gt = getFullName(shortname[:-4], dataroot)[:-4]+'_mask.png'
        pseudoLabel = getPseudoLabelPathFromShortName(shortname[:-4], label_path) 

        gt_image = np.array(Image.open(gt).split()[0])
        pseudoLabel_image = np.array(Image.open(pseudoLabel).split()[0])
        tumor_iou, bg_iou = metrics(pseudoLabel_image, gt_image)
        tumor.append(tumor_iou)
        bgList.append(bg_iou)

    average_bg = sum(bgList) / len(bgList)
    average_tm = sum(tumor) / len(tumor)
    print('average of background iou', average_bg)
    print('average of tumor iou', average_tm)

    print('miou:', (average_tm+average_bg)/2)

if __name__ == '__main__':
    label_path1 = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f04'
    label_path2 = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f045'
    label_path3 = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f05'
    label_path4 = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f055'
    label_path5 = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f06'

    # iou_labelcomparsion_p(label_path1)
    # iou_labelcomparsion_p(label_path2)
    # iou_labelcomparsion_p(label_path3)
    # iou_labelcomparsion_p(label_path4)
    # iou_labelcomparsion_p(label_path5)

    label1 = '/home/lintzuh@kean.edu/BUS/data/PseudoLabel'
    label2 = '/home/lintzuh@kean.edu/BUS/data/PseudoLabel_convexHull'

    iou_labelcomparsion(label1)
    iou_labelcomparsion(label2)


# %%
