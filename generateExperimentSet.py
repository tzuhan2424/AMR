from BUS.loadData import load
from help.helper import getPathShortName
class DataSetSeprator():
    def __init__(self, is_generate_train, is_generate_test,
                 is_generate_train_without_normal, is_generate_test_without_normal,
                 output_train_path='train_images_list.txt',
                 output_test_path='test_images_list.txt',
                 output_train_without_normal_path='train_images_without_normal_list.txt',
                 output_test_without_normal_path='test_images_without_normal_list.txt',
                 data_root='/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'):
        

        self.is_generate_train = is_generate_train
        self.is_generate_test = is_generate_test
        self.is_generate_train_without_normal = is_generate_train_without_normal
        self.is_generate_test_without_normal = is_generate_test_without_normal
        
         # Output paths
        self.output_train_path = output_train_path
        self.output_test_path = output_test_path
        self.output_train_without_normal_path = output_train_without_normal_path
        self.output_test_without_normal_path = output_test_without_normal_path

        
        self.data_root = data_root
    
    def getPathShortName(self,fullpath):
        import os
        directory, img_name = os.path.split(fullpath)
        subdirectories = directory.split('/')
        type = subdirectories[-1]

        return img_name, type
    def load(self, dataRoot):
        import os
        from sklearn.model_selection import train_test_split

        # load the image file
        benign_image = []
        # must skip the mask
        for i in os.listdir(os.path.join(dataRoot, 'benign')):
            if 'mask' not in i:
                benign_image.append(os.path.join(dataRoot, 'benign', i))
        malignant_image = []
        for i in os.listdir(os.path.join(dataRoot, 'malignant')):
            if 'mask' not in i:
                malignant_image.append(os.path.join(dataRoot, 'malignant', i))
        normal_image=[]
        for i in os.listdir(os.path.join(dataRoot, 'normal')):
            if 'mask' not in i:
                normal_image.append(os.path.join(dataRoot, 'normal', i))


        all_images = benign_image+malignant_image+normal_image
        all_labels = [1] * len(benign_image) + [1] * len(malignant_image) + [0] * len(normal_image)
        train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

    
        return train_images, train_labels, test_images, test_labels
    def writeFile(self, outputPath, arr):
        # file_path = 'train_images_list.txt'
        file_path = outputPath

        with open(file_path, 'w') as file:
            file.write('file name: (x, y, w, h)' + '\n')
            for item in arr:
                file.write(f"{item}\n")

    def generate_output(self):
        train_images, train_labels, test_images, test_labels = self.load(self.data_root)
        if self.is_generate_train:
            print('generate_train')
            tmp=[]
            for i, name in enumerate(train_images):
                shortName, type = self.getPathShortName(name)
                tmp.append(shortName[:-4]) 
            self.writeFile(self.output_train_path, tmp)
            
            print('finished')


        if self.is_generate_test:
            print('generate_test')

            tmp=[]
            for i, name in enumerate(test_images):
                shortName, type = self.getPathShortName(name)
                tmp.append(shortName[:-4]) 
            self.writeFile(self.output_test_path, tmp)

            print('finished')


        if self.is_generate_train_without_normal:
            print('generate_train_without_normal')

            tmp=[]
            for i, name in enumerate(train_images):
                shortName, type = self.getPathShortName(name)
                if type != 'normal':
                    tmp.append(shortName[:-4]) 


            self.writeFile(self.output_train_without_normal_path, tmp)
            print('finished')

        if self.is_generate_test_without_normal:
            print('generate_test_without_normal')

            tmp=[]
            for i, name in enumerate(test_images):
                shortName, type = self.getPathShortName(name)
                if type != 'normal':
                    tmp.append(shortName[:-4]) 


            self.writeFile(self.output_test_without_normal_path, tmp)
            print('finished')

    def read_file_to_list(self, file_path):
        with open(file_path, 'r') as file:
            # Skip the first line (e.g., a header)
            next(file)
            lines = file.readlines()
            return [line.strip() for line in lines]
        
    def summarizeTheList(self, outputPath):
        fileList= self.read_file_to_list(outputPath)
        # print(fileList)
        b = []
        m = []
        n = []

        for name in fileList:
            if name[0] == 'b':
                b.append(name)
            elif name[0] == 'm':
                m.append(name)
            elif name[0] == 'n':
                n.append(name)
            else:
                print('something wrong')
        # print('summarize of the dataset:', outputPath)
        # print('the whole list len: ', len(fileList))
        # print('benign', len(b))
        # print('malignant', len(m))
        # print('normal', len(n))


        # Display the results in a row-wise table format
        print('Summary of the dataset:', outputPath)
        print('-' * 45)
        print(f"|{'Benign':^10}|{'Malignant':^10}|{'Normal':^10}|{'Total':^10}|")
        print('-' * 45)
        print(f"|{len(b):^10}|{len(m):^10}|{len(n):^10}|{len(fileList):^10}|")
        print('-' * 45)
        



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
    # separateSet()
    s= DataSetSeprator(is_generate_train=True, 
                       is_generate_test=True,
                       is_generate_train_without_normal=True, 
                       is_generate_test_without_normal=True)
    # s.generate_output()
    s.summarizeTheList('record/train_images_list_0428.txt')
    s.summarizeTheList('record/test_images_list_0428.txt')
    s.summarizeTheList('record/train_images_without_normal_list_0428.txt')
    s.summarizeTheList('record/test_images_without_normal_list_0428.txt')
    