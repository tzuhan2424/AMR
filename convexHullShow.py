#%%
from convexHull import convexHullPic
def loadBenignMalignant():
    from help.camHelper import load
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

    return tumor



def convexHullViz(camFolder):
    import os
    import matplotlib.pyplot as plt

    tumor = loadBenignMalignant()


    processed_images = []

    j = 0
    for i, filename in enumerate(tumor):
        directory, img_name = os.path.split(filename)
        a, iou = convexHullPic(camFolder, filename)
        processed_images.append((a, img_name, iou))

        if (i + 1) % 4 == 0 or (i + 1) == len(tumor):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3)) # Adjust figsize as needed
            for ax, (img, name, iou) in zip(axes, processed_images):
                ax.imshow(img)
                ax.set_title(f"{name[:-4]}: {iou:.2f}")
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            processed_images = []
        # j+=1

        # if j==20:
        #     break





if __name__ == '__main__':
    convexHullViz('result/cam_merge')
# %%
