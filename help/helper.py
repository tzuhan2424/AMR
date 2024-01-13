def merge_rec_saveFile(filename, camfolder = 'cropCam', threshold=0.4, isSavefile=False, savePicDir='camTest'):
    """
    perform selective search from some cam folder
    return a rectangle 
    and could be able to save the rec with the original image file to viz purpose


    """
    import cv2
    import math
    import numpy as np
    import os


    # selective search
    shortname, type = getPathShortName(filename)
    image = cv2.imread(filename)


    # Create a Selective Search segmentation object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # Set the input image for segmentation
    ss.setBaseImage(image)
    # Perform the selective search segmentation
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    cam_path = os.path.join(camfolder, type, shortname[:-4] + '.npy')

    loaded_data = np.load(cam_path,allow_pickle=True).item()
    cam = loaded_data['high_res']
    cam = cam.reshape((image.shape[0], image.shape[1]))


   # Display the individual region segmentations
    def refineRec(rects, threshold):
        rects_refine = []
        for i, rect in enumerate(rects):
            x, y, w, h = rect
            roi = image[y:y + h, x:x + w]
            roi2 = cam[y:y + h, x:x + w]
            if h < image.shape[0]/2 and w < image.shape[1]/2:
                if np.mean(roi2) > threshold:
                    rects_refine.append(np.array([x, y, w, h]))
        return rects_refine

    rects_refine= refineRec(rects, threshold)
    while(len(rects_refine)== 0):
        threshold-=0.1
        print('decrease the threshold to ', threshold)
        rects_refine = refineRec(rects, threshold)

    new_X, new_Y  = math.inf, math.inf
    maxXpW= - math.inf
    maxYpH= - math.inf
    for i in range(len(rects_refine)):
        x, y, w, h = rects_refine[i]
        if x < new_X:
            new_X = x
        if y < new_Y:
            new_Y = y
        if x+w > maxXpW:
            maxXpW = x+w
        if y+h > maxYpH:
            maxYpH = y+h
    new_W = maxXpW-new_X
    new_H = maxYpH-new_Y


    # breakpoint()
    if isSavefile:
        cv2.rectangle(image, (new_X, new_Y), (new_X + new_W, new_Y + new_H), (0,255,0), 2)
        
        folder_path = './' +savePicDir+'/' + type
        os.makedirs(folder_path, exist_ok=True)
        fName = shortname[:-4] +'_merge'+ '.jpg'
        file_path = os.path.join(folder_path, fName) 
        cv2.imwrite(file_path, image)


        

    return new_X, new_Y, new_W, new_H

def boundingBoxFromGT(imageFullPath):
    """
    generate the bounding box from ground truth
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    # f ='benign (142)'
    # type = 'benign' if f[0] == 'b' else 'malignant'
    # filename = '../Dataset_BUSI_with_GT/' +type+ '/'+ f + '.png'
    # gtname = '../Dataset_BUSI_with_GT/' +type+ '/'+ f + '_mask.png'
    # print('f', f)
    # print('type', type)
    # print('filename!!: ', filename)
    # print('gtName!!:', gtname)


    MASKfullName = imageFullPath[:-4] + '_mask.png'



    # Load the ground truth label image (0 and 1 binary image)
    ground_truth_image = cv2.imread(MASKfullName, cv2.IMREAD_GRAYSCALE)

    if ground_truth_image is None:
        print("Error: Unable to load the ground truth image.")
        exit()

    # Find contours of the objects
    contours, _ = cv2.findContours(ground_truth_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create bounding boxes and draw them on the original image
    original_image = cv2.imread(imageFullPath)  # Load the corresponding original image


    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        # cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green bounding boxes

    # Convert BGR image to RGB for displaying with matplotlib
    # original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Display the original image with bounding boxes using matplotlib
    # plt.imshow(original_image_rgb)
    # plt.axis('off')  # Hide axis ticks and labels
    # plt.savefig('test.png', bbox_inches='tight', pad_inches=0)

    return bounding_boxes # because we could have multiple label in picture, so our bounding box is a list

def merge_bounding_boxes(bounding_boxes):
    if not bounding_boxes:
        return None

    # Initialize the merged bounding box coordinates with the first box
    x_min, y_min, width_max, height_max = bounding_boxes[0]
    x_max = x_min + width_max
    y_max = y_min + height_max

    # Loop through the remaining bounding boxes and update the merged coordinates
    for box in bounding_boxes[1:]:
        x, y, width, height = box
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + width)
        y_max = max(y_max, y + height)

    # Compute the final width and height of the merged bounding box
    width_merged = x_max - x_min
    height_merged = y_max - y_min

    # Return the coordinates of the merged bounding box
    return x_min, y_min, width_merged, height_merged



def rectangle_intersection_area(rect1, rect2):
    # Calculate the coordinates of the intersection rectangle
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])

    # Check if there's a valid intersection
    if x_right < x_left or y_bottom < y_top:
        return 0

    # Calculate the intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def rectangle_union_area(rect1, rect2):
    # Calculate the areas of the individual rectangles
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    # Calculate the union area by adding the areas of both rectangles
    union_area = area_rect1 + area_rect2

    # Subtract the intersection area to avoid double-counting
    union_area -= rectangle_intersection_area(rect1, rect2)
    return union_area
def rectangle_iou(rect1, rect2):
    """
    # top-left and bottom-right coordinates (x1, y1, x2, y2)
    """
    intersection_area = rectangle_intersection_area(rect1, rect2)
    union_area = rectangle_union_area(rect1, rect2)

    # Handle the case of division by zero to avoid errors
    if union_area == 0:
        return 0

    iou = intersection_area / union_area
    return iou



def getPathShortName(fullpath):
    import os
    directory, img_name = os.path.split(fullpath)
    subdirectories = directory.split('/')
    type = subdirectories[-1]

    return img_name, type


def getFullName(name, dataroot):
    import os
    if name[0] == 'b':
        type = 'benign'
    elif name[0] == 'm':
        type = 'malignant'
    elif name[0] == 'n':
        type = 'normal'

    fullPath = os.path.join(dataroot, type, name+'.png')

    return fullPath



    


