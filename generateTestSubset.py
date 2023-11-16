from BUS.loadData import load
dataRoot = '/home/lintzuh@kean.edu/BUS/data/Dataset_BUSI_with_GT'
train_images, train_labels, test_images, test_labels = load(dataRoot)

benign_test_image = []
malignant_test_image = []

for item in test_images:
    if 'benign' in item:
        benign_test_image.append(item)
    elif 'malignant' in item:
        malignant_test_image.append(item)

testSet = benign_test_image+malignant_test_image
print(testSet[:3])
print(len(benign_test_image))
print(len(malignant_test_image))


output_file_path = '/home/lintzuh@kean.edu/BUS/AMR/record/testSetWithoutNormal.txt'

# Writing each file path to the text file, one line at a time
with open(output_file_path, 'w') as file:
    for path in testSet:
        file.write(path + '\n')

