#%%
def camPic(name, camFolder ='result/cam_merge'):
    import os
    from help.helper import getFullName
    from help.camHelper import visualizeCam
    import matplotlib.pyplot as plt

    
    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    fullName = getFullName(name, dataRoot)



    directory, img_name = os.path.split(fullName)
    a = visualizeCam(camFolder, fullName)
    plt.imshow(a)
    plt.title(name)

    plt.axis('off') # to turn off the axis
    plt.show()

def vex(name, camFolder ='result/cam_merge'):
    import os
    from help.helper import getFullName
    from help.camHelper import visualizeCam
    import matplotlib.pyplot as plt
    from convexHull import convexHullPic


    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    fullName = getFullName(name, dataRoot)

    a, iou = convexHullPic(camFolder, fullName)

    plt.imshow(a)
    plt.title(f"{name}: {iou:.2f}")

    plt.axis('off') # to turn off the axis
    plt.show()

def gt(name):
    from help.helper import getFullName
    import cv2
    import matplotlib.pyplot as plt

    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    
    fullName = getFullName(name, dataRoot)
    a = cv2.imread(fullName[:-4] + '_mask.png')
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for displaying in Matplotlib
    label_mask = a[:, :, 0] > 0
    # Apply this mask to all channels to turn those areas white
    a[label_mask] = [255, 255, 255] 


    plt.imshow(a)
    plt.title(f"{name}")

    plt.axis('off') # to turn off the axis
    plt.show()



def pseudoLabel(name):
    from help.helper import getFullName
    import cv2
    import matplotlib.pyplot as plt

    dataRoot = '/home/lintzuh@kean.edu/BUS/data/PseudoLabel_convexHull'
    
    fullName = getFullName(name, dataRoot)


    a = cv2.imread(fullName)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for displaying in Matplotlib
    label_mask = a[:, :, 0] > 0
    # Apply this mask to all channels to turn those areas white
    a[label_mask] = [255, 255, 255] 


    plt.imshow(a)
    plt.title(f"{name}")

    plt.axis('off') # to turn off the axis
    plt.show()


    



    









if __name__ == '__main__':
    candidates = ['benign (111)', 'benign (126)', 'benign (130)', 'malignant (54)', 'benign (130)', 'malignant (194)', 'benign (388)']


    for can in candidates:
        camPic(can)
        vex(can)
        gt(can)
        pseudoLabel(can)

# %%
