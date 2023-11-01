import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from help.helper import merge_rec_saveFile, boundingBoxFromGT,merge_bounding_boxes,rectangle_iou
from BUS.loadData import load
from help.helper import getPathShortName


def visualizeCamOriginal(camFolder, filename):
    """
    use the file in the camfolder to visualize in the same picture
    # filename = 'benign (162)'
    return pic

    this is only used the grade cam map we generated long before
    """
    directory, img_name = os.path.split(filename)

    # Split the directory into subdirectories
    subdirectories = directory.split('/')# 0->root, 1->datasetsomething, 2->benign
    
    type = subdirectories[2] #benign



    image = cv2.imread(filename)
    cam = np.load(camFolder + '/' +type + '/' + img_name[:-4] + '.npy')
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    cam = np.uint8(cam)

    cam_image = cv2.applyColorMap(cam, cv2.COLORMAP_JET) 
    # Overlay the CAM on the original image

    cam_image = cv2.addWeighted(image,  0.8, cam_image, 0.4, 0)
    
    cam_image_rgb = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(cam_image_rgb)  # Use 'cmap=None' for color images
    # plt.axis('off')  # Turn off axis ticks and labels
    # plt.show() 

    return cam_image_rgb
def visualizeCam(camFolder, filename):
    """
    use the file in the camfolder to visualize in the same picture
    return pic
    """

    img_name, type = getPathShortName(filename)
    image = cv2.imread(filename)
    cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')

    loaded_data = np.load(cam_path,allow_pickle=True).item()
    cam = loaded_data['high_res']
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    cam = np.uint8(cam)

    cam = cam.reshape((image.shape[0], image.shape[1]))
    cam_image = cv2.applyColorMap(cam, cv2.COLORMAP_JET) 
    # Overlay the CAM on the original image

    cam_image = cv2.addWeighted(image,  0.8, cam_image, 0.4, 0)
    
    cam_image_rgb = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(cam_image_rgb)  # Use 'cmap=None' for color images
    # plt.axis('off')  # Turn off axis ticks and labels
    # plt.show() 
    return cam_image_rgb




def compareGTandCamROI(cam, isShowPic = False, threshold = 0.6, isSaveROI=False):

    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    train_images, train_labels, test_images, test_labels = load(dataRoot)


    benign_image = []
    malignant_image = []

    for item in train_images:
        if 'benign' in item:
            benign_image.append(item)
        elif 'malignant' in item:
            malignant_image.append(item)

    image_set = benign_image+malignant_image

   
    benign_iou=0
    malignant_iou = 0

    if isShowPic:
        images_per_row = 4
        num_rows = (len(image_set) + images_per_row - 1) // images_per_row
        num_subplots = num_rows * images_per_row
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(12, 4 * num_rows))  

    ROI_list = []
    valid_ROI_list = []
    for i, name in enumerate(image_set):
       

        #name is full path, camfolder = 'result/cams'
        x, y, w, h = merge_rec_saveFile(name, camfolder=cam, threshold=threshold, isSavefile=False)
        shortName, type = getPathShortName(name)

        ROI_list.append((shortName[:-4], (x, y, w, h))) 

        gt_x, gt_y, gt_w, gt_h = merge_bounding_boxes(boundingBoxFromGT(name))       
        iou = rectangle_iou([x, y, x+w, y+h],[gt_x, gt_y, gt_x+gt_w, gt_y+gt_h])

        if iou > 0.6:
            valid_ROI_list.append((shortName[:-4], (x, y, w, h), iou)) 


        #caculate iou
        if type == 'benign':
            benign_iou+=iou
        elif type == 'malignant':
            malignant_iou+=iou


        if isShowPic:
            row_idx = i // images_per_row
            col_idx = i % images_per_row
            image = cv2.imread(name)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.rectangle(image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0,0,255), 2)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[row_idx, col_idx].imshow(image_rgb)
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].text(0.5, 1.05, f"{shortName[:-4]}, iou = {iou: .2f}", ha='center', va='center',
                                transform=axes[row_idx, col_idx].transAxes, fontsize=12, color='black')

        
    if isShowPic:
        for i in range(len(image_set), num_subplots):
            row_idx = i // images_per_row
            col_idx = i % images_per_row
            axes[row_idx, col_idx].axis('off') 

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.4)
        plt.tight_layout()
        plt.show()

    if isSaveROI:
        file_path = 'ROI_FULLSET.txt'
        with open(file_path, 'w') as file:
            # Iterate over the list and write each item followed by a newline character
            file.write('file name: (x, y, w, h)' + '\n')
            for item in ROI_list:
                file.write(f"{item[0]}, {item[1]}\n")


        valid_ROI_file_path = 'ROI_validSET.txt'
        with open(valid_ROI_file_path, 'w') as file:
            # Iterate over the list and write each item followed by a newline character
            file.write('file name: (x, y, w, h), iou' + '\n')
            for item in valid_ROI_list:
                file.write(f"{item[0]}, {item[1]}, {item[2]}\n")


    print('benign miou:', benign_iou/len(benign_image))
    print('malignant miou:', malignant_iou/len(malignant_image))
    print('mIOU:', (benign_iou+malignant_iou)/(len(benign_image)+len(malignant_image)))


def compareDifferentCam(first, second, third, fourth, howmany=5):
    """
    this can compare four cam in one row
    """

    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    train_images, train_labels, test_images, test_labels = load(dataRoot)

    benign_image = []
    malignant_image = []

    for item in train_images:
        if 'benign' in item:
            benign_image.append(item)
        elif 'malignant' in item:
            malignant_image.append(item)

    tumor = benign_image+malignant_image

    
    j = 0
    for filename in tumor:
        directory, img_name = os.path.split(filename)

        # 'result/cam_merge' 'result/cam_spotlight' 'result_previous/cams' 'result_previous/cams_spotlight'

        a = visualizeCam(first, filename)
        b = visualizeCam(second, filename)

        c = visualizeCam(third, filename)
        d = visualizeCam(fourth, filename)
 
        
        
        
        fig, axes = plt.subplots(1, 4, figsize=(9, 4))
        # Plot each image on a subplot
        axes[0].imshow(a)
        axes[0].set_title(first+ img_name[:-4])
        axes[0].axis('off')

        axes[1].imshow(b)
        axes[1].set_title(second)
        axes[1].axis('off')

        axes[2].imshow(c)
        axes[2].set_title(third)
        axes[2].axis('off')

        axes[3].imshow(d)
        axes[3].set_title(fourth)
        axes[3].axis('off')

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        
        j+=1
        if j == howmany:
            break