#%%
# the one that cover the whole tumor #benign 126
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from help.helper import merge_rec_saveFile, boundingBoxFromGT,merge_bounding_boxes,rectangle_iou
from BUS.loadData import load
from help.helper import getPathShortName
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

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

def readFromSet_list(file_path = '/home/lintzuh@kean.edu/BUS/AMR/record/ROI_FULLSET.txt'):
    import re
    import sys
   
    
    sys.path.append('/home/lintzuh@kean.edu/BUS/mean-teacher-medical-imaging/code')
    def getFullPathName(dataRoot, shortName):
        # input: benign (126)
        import os
        if shortName[0] == 'b':
            type = 'benign'
        elif shortName[0] == 'm':
            type = 'malignant'
        else:
            type = 'normal'

        Fp = os.path.join(dataRoot, type, shortName+'.png')
        return Fp
    imageName = []
    imageNameRec=[]
    dataroot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'

    with open(file_path, 'r') as file:
        next(file)

        for line in file:
            # Regular expression pattern to match the required fields
            pattern = r'([^,]+ \(\d+\)), \((\d+, \d+, \d+, \d+)\)'

            match = re.match(pattern, line)

            if match:
                filename = match.group(1)
                coordinates = tuple(map(int, match.group(2).split(', ')))
                


                fullFileName =getFullPathName(dataroot, filename)
                imageName.append(fullFileName)
                imageNameRec.append(coordinates)
            else:
                print('dangerous, when reading file')
    return imageName, imageNameRec


def plot_histogram(image):
    """Plot the histogram of the given image."""
    plt.hist(image.ravel(), bins=256, range=[0, 255])
    plt.title('Histogram of the Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def count_pixels_in_range(image, min_value, max_value):
    """Count the number of pixels in the specified range."""
    return np.sum((image >= min_value) & (image <= max_value))



def find_points_in_range(image, min_value, max_value):
    """Find the coordinates of pixels in the specified range."""
    y_coords, x_coords = np.where((image >= min_value) & (image <= max_value))
    return np.column_stack((x_coords, y_coords))


def compute_convex_hull(points):
    """Compute the convex hull of a set of points."""
    return ConvexHull(points)

def plot_convex_hull(image, hull, points):
    """Plot the image and the convex hull."""
    plt.imshow(image, cmap='gray')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-')
    plt.show()


def find_percent_threshold(image, percent):
    """
    Find a threshold value in the image such that the number of pixels above this 
    threshold is 20% of the total number of pixels.

    :param image: The processed image.
    :return: The threshold value.
    """
    # Calculate histogram
    hist, bins = np.histogram(image, bins=256, range=(0, 255))

    # Compute the cumulative sum from the end
    cum_sum = np.cumsum(hist[::-1])[::-1]

    # Total number of pixels
    total_pixels = image.size

    # Find the threshold index
    threshold_index = np.where(cum_sum <= total_pixels * percent)[0][0]

    # Corresponding bin value
    threshold_value = bins[threshold_index]

    return threshold_value

def plot_convex_hulls(image, points1, points2, color1='r', color2='b'):
    """
    Plot the image and the convex hulls for two sets of points.

    :param image: The image on which to plot the convex hulls.
    :param points1: The first set of points.
    :param points2: The second set of points.
    :param color1: The color for the first convex hull.
    :param color2: The color for the second convex hull.
    """
    plt.imshow(image, cmap='gray')

    # Compute and plot the first convex hull
    if len(points1) > 2:  # Convex hull requires at least 3 points
        hull1 = ConvexHull(points1)
        for simplex in hull1.simplices:
            # x = points1[simplex, 0]
            # y = points1[simplex, 1]
            plt.plot(points1[simplex, 0], points1[simplex, 1], color1 + '-')

    # Compute and plot the second convex hull
    if len(points2) > 2:  # Convex hull requires at least 3 points
        hull2 = ConvexHull(points2)
        for simplex in hull2.simplices:
            plt.plot(points2[simplex, 0], points2[simplex, 1], color2 + '-')

    plt.show()


def find_extreme_points(points):
    if len(points) > 2:  # Convex hull requires at least 3 points
        hull = ConvexHull(points)

        # Extract the hull points
        hull_points = points[hull.vertices, :]

        # Find the leftmost, rightmost, top, and bottom points
        x_leftMost = np.min(hull_points[:, 0])
        x_rightMost = np.max(hull_points[:, 0])
        y_top = np.min(hull_points[:, 1])
        y_bottom = np.max(hull_points[:, 1])

        return x_leftMost, x_rightMost, y_top, y_bottom

    else:
        return None, None, None, None
    




def plot_with_bounding_box(image, points, bounding_box):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    """
    Plot the image with a bounding box.

    :param image: The image to plot.
    :param points: The points used to create the bounding box.
    :param bounding_box: The (x_left, x_right, y_top, y_bottom) bounding box coordinates.
    """
    x_left, x_right, y_top, y_bottom = bounding_box

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Draw the points (optional)
    # ax.scatter(points[:, 0], points[:, 1], c='red')

    # Draw the bounding box
    # Note: Rectangle takes (x, y) of the lower-left corner, width, and height
    rect = Rectangle((x_left, y_top), x_right - x_left, y_bottom - y_top,
                     edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    plt.show()


def plot_with_bounding_boxes(image, points1, points2, bounding_box1, bounding_box2):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x_left1, x_right1, y_top1, y_bottom1 = bounding_box1
    x_left2, x_right2, y_top2, y_bottom2 = bounding_box2

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Draw the first bounding box
    rect1 = Rectangle((x_left1, y_top1), x_right1 - x_left1, y_bottom1 - y_top1,
                     edgecolor='red', facecolor='none')
    ax.add_patch(rect1)

    # Draw the second bounding box
    rect2 = Rectangle((x_left2, y_top2), x_right2 - x_left2, y_bottom2 - y_top2,
                     edgecolor='blue', facecolor='none')
    ax.add_patch(rect2)

    plt.show()
def draw_bounding_boxes(image, bounding_box1, bounding_box2):
    x_left1, x_right1, y_top1, y_bottom1 = bounding_box1
    x_left2, x_right2, y_top2, y_bottom2 = bounding_box2

    # Draw the first bounding box in red
    image_with_boxes = cv2.rectangle(image.copy(), (x_left1, y_top1), (x_right1, y_bottom1), (255, 0, 0), 2)

    # Draw the second bounding box in blue
    image_with_boxes = cv2.rectangle(image_with_boxes, (x_left2, y_top2), (x_right2, y_bottom2), (0, 0, 255), 2)

    return image_with_boxes

def return_plot_with_bounding_boxes(image, points1, points2, bounding_box1, bounding_box2):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x_left1, x_right1, y_top1, y_bottom1 = bounding_box1
    x_left2, x_right2, y_top2, y_bottom2 = bounding_box2

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Draw the first bounding box
    rect1 = Rectangle((x_left1, y_top1), x_right1 - x_left1, y_bottom1 - y_top1,
                     edgecolor='red', facecolor='none')
    ax.add_patch(rect1)

    # Draw the second bounding box
    rect2 = Rectangle((x_left2, y_top2), x_right2 - x_left2, y_bottom2 - y_top2,
                     edgecolor='blue', facecolor='none')
    ax.add_patch(rect2)



     # Render the figure to a canvas
    fig.canvas.draw()

    # Convert the figure to a NumPy array
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # Close the figure to free memory

    return img_data





    
# def getBoundingBox(camFolder, filename):
#     img_name, type = getPathShortName(filename)
#     print(img_name[:-4])
#     image = cv2.imread(filename)
#     cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')

#     loaded_data = np.load(cam_path,allow_pickle=True).item()
#     cam = loaded_data['high_res']
#     cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
#     cam = np.uint8(cam)
#     cam = cam.reshape((image.shape[0], image.shape[1]))

#     num=find_percent_threshold(cam, 0.025)
#     # print('num', num)
#     s1=count_pixels_in_range(cam, num, 255)
#     print('s1',s1)
#     points1 = find_points_in_range(cam, num, 250)

def determine_optimal_clusters(points, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(points)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph to observe 'The elbow'
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Ask user to determine the elbow point
    print("Please observe the plot and enter the number of clusters at the elbow point.")
    n_clusters = int(input("Enter the optimal number of clusters: "))
    return n_clusters

def plot_clustered_convex_hulls(image, points, n_clusters, colors):
    # Apply K-means clustering with the determined number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    labels = kmeans.labels_

    plt.imshow(image, cmap='gray')

    # Compute and plot the convex hull for each cluster
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        if len(cluster_points) > 2:  # Convex hull requires at least 3 points
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], colors[i] + '-')

    plt.show()



def convexHullPic(camFolder, filename):
    img_name, type = getPathShortName(filename)
    # print(img_name[:-4])
    image = cv2.imread(filename)
    cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')

    loaded_data = np.load(cam_path,allow_pickle=True).item()
    cam = loaded_data['high_res']
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    cam = np.uint8(cam)
    cam = cam.reshape((image.shape[0], image.shape[1]))


    num=find_percent_threshold(cam, 0.025)
    points1 = find_points_in_range(cam, num, 250)
    bounding_box1 = find_extreme_points(points1)



    num2=find_percent_threshold(cam, 0.01)
    points2 = find_points_in_range(cam, num2, 250)
    bounding_box2 = find_extreme_points(points2)
    
    # Check if bounding boxes are not None and contain valid coordinates
    if bounding_box1 and bounding_box2 and all(bounding_box1) and all(bounding_box2):
        x1_leftMost, x1_rightMost, y1_top, y1_bottom = bounding_box1
        x2_leftMost, x2_rightMost, y2_top, y2_bottom = bounding_box2

        iou = rectangle_iou([x1_leftMost, y1_top, x1_rightMost, y1_bottom], [x2_leftMost, y2_top, x2_rightMost, y2_bottom])
        pic = draw_bounding_boxes(image, bounding_box1, bounding_box2)
    else:
        print(f"Invalid bounding box data for image {img_name}.")
        # Handle the case where bounding boxes are invalid
        iou = 0
        pic = image 

    return pic, iou

def convexHullPic_singleBbox(camFolder, filename):
    img_name, type = getPathShortName(filename)
    # print(img_name[:-4])
    image = cv2.imread(filename)
    cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')

    loaded_data = np.load(cam_path,allow_pickle=True).item()
    cam = loaded_data['high_res']
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    cam = np.uint8(cam)
    cam = cam.reshape((image.shape[0], image.shape[1]))


    num=find_percent_threshold(cam, 0.025)
    points1 = find_points_in_range(cam, num, 250)
    bounding_box1 = find_extreme_points(points1)
    # print('points1')
    # print('bounding_box1', bounding_box1)


     # image_with_boxes = cv2.rectangle(image.copy(), (x_left1, y_top1), (x_right1, y_bottom1), (255, 0, 0), 2)

    # Check if bounding boxes are not None and contain valid coordinates
    if bounding_box1:
        # Draw the bounding box on the image
        x_leftMost, x_rightMost, y_top, y_bottom = bounding_box1
        cv2.rectangle(image, (x_leftMost, y_top), (x_rightMost, y_bottom), (0, 255, 0), 2)
        pic = image
    else:
        print(f"Invalid bounding box data for image {img_name}.")
        pic = image 

    return pic


def getDot(camFolder, filename, isShow=False):
    """
    use the file in the camfolder to visualize in the same picture
    return pic
    """

    img_name, type = getPathShortName(filename)
    # print(img_name[:-4])
    image = cv2.imread(filename)
    cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')

    loaded_data = np.load(cam_path,allow_pickle=True).item()
    cam = loaded_data['high_res']
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    cam = np.uint8(cam)
    cam = cam.reshape((image.shape[0], image.shape[1]))


    # version2
    # num=find_percent_threshold(cam, 0.025)
    # # print('num', num)
    # # s=count_pixels_in_range(cam, num, 255)
    # # print('s',s)
    # points1 = find_points_in_range(cam, num, 250)

    # max_clusters = 3
    # n_clusters = determine_optimal_clusters(points1, max_clusters)
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # Extend this list based on max_clusters
    # plot_clustered_convex_hulls(image, points1, n_clusters, colors[:n_clusters])


    # version 1
    # plot_histogram(cam)
    num=find_percent_threshold(cam, 0.025)
    # print('num', num)
    s1=count_pixels_in_range(cam, num, 255)
    # print('s1',s1)
    points1 = find_points_in_range(cam, num, 250)
        # Compute the convex hull
    # hull = compute_convex_hull(points)
    # Plot the image and the convex hull
    # plot_convex_hull(image, hull, points)


    bounding_box1 = find_extreme_points(points1)
    # plot_with_bounding_box(image, points1, bounding_box1)






    num2=find_percent_threshold(cam, 0.01)
    # print('num', num)
    s2=count_pixels_in_range(cam, num2, 255)
    # print('s2',s2)
    points2 = find_points_in_range(cam, num2, 250)
    bounding_box2 = find_extreme_points(points2)
    # plot_with_bounding_box(image, points2, bounding_box2)


    # plot_with_bounding_boxes(image, points1 ,points2, bounding_box1,bounding_box2)

    # plot_convex_hulls(image, points1, points2)

    x1_leftMost, x1_rightMost, y1_top, y1_bottom = bounding_box1
    x2_leftMost, x2_rightMost, y2_top, y2_bottom = bounding_box2

    iou = rectangle_iou([x1_leftMost, y1_top, x1_rightMost, y1_bottom],[x2_leftMost, y2_top, x2_rightMost, y2_bottom])
    
    if iou > 0.7:
        # print(img_name[:-4])
        if isShow:
            plot_with_bounding_boxes(image, points1 ,points2, bounding_box1,bounding_box2)
        # return 1, (x1_leftMost, y1_top, x1_rightMost-x1_leftMost,y1_bottom-y1_top)
        return 1
    return 0


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


def convexHullGetWholeTrainingSetLabel(threshold = 0.7, train_path='/home/lintzuh@kean.edu/BUS/AMR/record/ROI_FULLSET.txt', camFolder='result/cam_merge', isSaveROI=False, savedROI_filename='convexHull_ROI_theWholeTrainingSet.txt'):
    train_set, _ = readFromSet_list(train_path)
    ROI_list = []
    for fullName in train_set:
        #fullName = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT/benign/benign (270).png'
        img_name, type = getPathShortName(fullName) #'benign (270).png'
        print(img_name)

        image = cv2.imread(fullName)
        cam_path=os.path.join(camFolder, type, img_name[:-4] + '.npy')
        loaded_data = np.load(cam_path,allow_pickle=True).item()
        cam = loaded_data['high_res']
        cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
        cam = np.uint8(cam)
        cam = cam.reshape((image.shape[0], image.shape[1]))

    
        num=find_percent_threshold(cam, 0.025)
        points1 = find_points_in_range(cam, num, 250)
        bounding_box1 = find_extreme_points(points1)

        x1_leftMost, x1_rightMost, y1_top, y1_bottom = bounding_box1
        # x,y,w,h = x1_leftMost, y1_top, x1_rightMost-x1_leftMost,y1_bottom-y1_top

        #see exception
        w = None
        h = None
        if x1_rightMost is None or x1_leftMost is None:
            print(f"{img_name}, Error: Rightmost or leftmost x-coordinate is None")
        else:
            w = x1_rightMost - x1_leftMost

        if y1_bottom is None or y1_top is None:
            print(f"{img_name}, Error: Bottom or top y-coordinate is None")
        else:
            h = y1_bottom - y1_top

        # Check if w and h have been set properly before assigning x, y, w, h
        if w is not None and h is not None:
            x, y = x1_leftMost, y1_top
        else:
            # Handle the case where w or h couldn't be computed due to None values
            print(f"{img_name}, Error: Cannot compute width or height due to None values")
            # Set x, y, w, h to None or some default values as appropriate
            x = y = w = h = None

        ROI_list.append((img_name[:-4], (x, y, w, h)))   
    
    if isSaveROI:
        file_path = savedROI_filename
        with open(file_path, 'w') as file:
            # Iterate over the list and write each item followed by a newline character
            file.write('file name: (x, y, w, h)' + '\n')
            for item in ROI_list:
                file.write(f"{item[0]}, {item[1]}\n")

if __name__ == '__main__':
    pass
    # convexHullGetWholeTrainingSetLabel(camFolder="/data/lintzuh/BUS/Swin-Transformer/gradCamResult", isSaveROI=True, savedROI_filename = 'convexHull_ROI_theWholeTrainingSet_swin.txt')
    # convexHullGetValid(isSaveROI=True)


    # dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    # filename = 'benign (126).png'
    # fullName = os.path.join(dataRoot,'benign' ,filename)
    # print(fullName)

    # getDot('result/cam_merge', fullName)

    # # just edge benign 186
    # filename = 'benign (186).png'
    # fullName = os.path.join(dataRoot,'benign' ,filename)
    # getDot('result/cam_merge', fullName)


    # file name: (x, y, w, h)
    # benign (270), (275, 25, 135, 67)
    # train_path = '/home/lintzuh@kean.edu/BUS/AMR/record/ROI_FULLSET.txt'
    # train_set, _ = readFromSet_list(train_path)
    


    # dataroot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'


    # st = []
    # for fullName in train_set:
    #     v = getDot('result/cam_merge', fullName)
    #     if v == 1:
    #         img_name, type = getPathShortName(fullName)
    #         st.append(img_name)
    #     # i+=1
    #     # if i == 20:
    #     #     break

    # print('len(st)', len(st))
    # print(st)

    # for fullName in select:
    #     fullName = os.path.join(dataroot, 'benign', fullName+'.png')
    #     getDot('result/cam_merge', fullName)
    #     i+=1
    #     if i == 20:
    #         break






# %%
