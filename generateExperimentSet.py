from BUS.loadData import load
from help.helper import getPathShortName

def separateSet():
    dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
    train_images, train_labels, test_images, test_labels = load(dataRoot)

    train_ROI_list = []
    b=[]
    n=[]
    m=[]
    for i, name in enumerate(train_images):
        shortName, type = getPathShortName(name)
        train_ROI_list.append(shortName[:-4]) 
        # if type == 'benign':
        #     b.append(shortName[:-4])
        # if type == 'normal':
        #     n.append(shortName[:-4])
        # if type == 'malignant':
        #     m.append(shortName[:-4])

    test_list = []
    for i, name in enumerate(test_images):
        shortName, type = getPathShortName(name)
        test_list.append(shortName[:-4]) 

        # if type == 'benign':
        #     b.append(shortName[:-4])
        # if type == 'normal':
        #     n.append(shortName[:-4])
        # if type == 'malignant':
        #     m.append(shortName[:-4])
    
    # print(len(b))
    # print(len(n))
    # print(len(m))
    # print(len(b)+len(n)+len(m))
    # print(ROI_list[:5])
    # print(b[:10])
    # print(m[:10])   

    # file_path = 'train_images_list.txt'
    # with open(file_path, 'w') as file:
    #     # Iterate over the list and write each item followed by a newline character
    #     file.write('file name: (x, y, w, h)' + '\n')
    #     for item in train_ROI_list:
    #         file.write(f"{item}\n")

    

    # test_file_path = 'test_images_list.txt'
    # with open(test_file_path, 'w') as file:
    #     # Iterate over the list and write each item followed by a newline character
    #     file.write('file name: (x, y, w, h)' + '\n')
    #     for item in test_list:
    #         file.write(f"{item}\n")





if __name__ == '__main__':
    separateSet()