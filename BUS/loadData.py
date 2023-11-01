# def load():
#     import os
#     from sklearn.model_selection import train_test_split

#     # load the image file
#     benign_image = []
#     # must skip the mask
#     for i in os.listdir("../Dataset_BUSI_with_GT/benign/"):
#         if 'mask' not in i:
#             benign_image.append("../Dataset_BUSI_with_GT/benign/" + i)
#     malignant_image = []
#     for i in os.listdir("../Dataset_BUSI_with_GT/malignant/"):
#         if 'mask' not in i:
#             malignant_image.append("../Dataset_BUSI_with_GT/malignant/" + i)
#     normal_image=[]
#     for i in os.listdir("../Dataset_BUSI_with_GT/normal/"):
#         if 'mask' not in i:
#             normal_image.append("../Dataset_BUSI_with_GT/normal/" + i)


#     all_images = benign_image+malignant_image+normal_image
#     all_labels = [1] * len(benign_image) + [1] * len(malignant_image) + [0] * len(normal_image)

#     # all_images = benign_image+malignant_image
#     # all_labels = [1] * len(benign_image) + [0] * len(malignant_image)
#     train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

   
#     return train_images, train_labels, test_images, test_labels


def load(dataRoot):
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


def saveDataset(my_list, file_name = "my_list.txt"):
    with open(file_name, "w") as file:
    # Iterate through the list and write each item to the file
        for item in my_list:
            file.write(item + "\n")


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels=load('../data/Dataset_BUSI_with_GT')
    print(train_images[:5])
    print(train_labels[:5])
    print(test_images[:5])
    print(test_labels[:5])
    print('train',len(train_images))
    print('test', len(test_images))


    saveDataset(train_images, file_name='trainSet')
    saveDataset(test_images, file_name='testSet')


