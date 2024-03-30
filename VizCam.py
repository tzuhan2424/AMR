#%%
def camViz(camFolder):
    import random

    from help.camHelper import visualizeCam, load
    import os
    import matplotlib.pyplot as plt
    
    random.seed(42)

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
    selected_images = random.sample(tumor, int(len(tumor) * 0.10))
    processed_images = []

    for i, filename in enumerate(selected_images):
        directory, img_name = os.path.split(filename)
        a = visualizeCam(camFolder, filename)
        processed_images.append((a, img_name))

        if (i + 1) % 4 == 0 or (i + 1) == len(selected_images):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3)) # Adjust figsize as needed
            for ax, (img, name) in zip(axes, processed_images):
                ax.imshow(img)
                ax.set_title(name[:-4])
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            processed_images = []




#%%
if __name__ == '__main__':
    # p='/home/lintzuh@kean.edu/BUS/ReCAM/result_default5/cam_mask'
    p='/home/lintzuh@kean.edu/BUS/ReCAM/result_default5/recam_mask'
    p='/data/lintzuh/BUS/Swin-Transformer/gradCamResult'
    camViz(p)
# %%
