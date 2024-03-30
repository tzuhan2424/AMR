import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from help.helper import merge_rec_saveFile, boundingBoxFromGT,merge_bounding_boxes,rectangle_iou
from BUS.loadData import load
from help.helper import getPathShortName
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from convexHull import readFromSet_list, find_extreme_points, find_percent_threshold, find_points_in_range

def convexHullGetValid(threshold = 0.7, train_path='/home/lintzuh@kean.edu/BUS/AMR/record/ROI_FULLSET.txt', camFolder='result/cam_merge', isSaveROI=False):
    train_set, _ = readFromSet_list(train_path)

    ROI_list = []
    for fullName in train_set:
        #fullName = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT/benign/benign (270).png'
        img_name, type = getPathShortName(fullName) #'benign (270).png'
        image = cv2.imread(fullName)
        cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')
        loaded_data = np.load(cam_path,allow_pickle=True).item()
        cam = loaded_data['high_res']
        cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
        cam = np.uint8(cam)
        cam = cam.reshape((image.shape[0], image.shape[1]))
        return 

    
        num=find_percent_threshold(cam, 0.025)
        points1 = find_points_in_range(cam, num, 250)
        bounding_box1 = find_extreme_points(points1)

        num2=find_percent_threshold(cam, 0.01)
        points2 = find_points_in_range(cam, num2, 250)
        bounding_box2 = find_extreme_points(points2)

        x1_leftMost, x1_rightMost, y1_top, y1_bottom = bounding_box1
        x2_leftMost, x2_rightMost, y2_top, y2_bottom = bounding_box2

        iou = rectangle_iou([x1_leftMost, y1_top, x1_rightMost, y1_bottom],[x2_leftMost, y2_top, x2_rightMost, y2_bottom])
        x,y,w,h = x1_leftMost, y1_top, x1_rightMost-x1_leftMost,y1_bottom-y1_top

        if iou > threshold:
            ROI_list.append((img_name[:-4], (x, y, w, h)))
    
    if isSaveROI:
        file_path = 'convexHull_ROI.txt'
        with open(file_path, 'w') as file:
            # Iterate over the list and write each item followed by a newline character
            file.write('file name: (x, y, w, h)' + '\n')
            for item in ROI_list:
                file.write(f"{item[0]}, {item[1]}\n")

if __name__ == '__main__':
    convexHullGetValid()