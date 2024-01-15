#%%
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Replace 'path_to_your_image.png' with the path to your label image
image_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f045/benign/benign (1)_palette.png'
# image_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amr_f06b03/benign/benign (1).png' #has 0, 1, 255
# image_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label/benign/benign (1)_palette.png' #only has 0, and 1
# Open the image
image = Image.open(image_path)

# Convert the image to grayscale (assuming it's a single-channel image)
# image = image.convert('L')

# Read pixel values
pixel_values = list(image.getdata())
# Count the frequency of each pixel value
pixel_count = Counter(pixel_values)

# Print the result
print(pixel_count)



mask = Image.open(image_path).split()[0]
pixel_values = list(image.getdata())
# Count the frequency of each pixel value
pixel_count = Counter(pixel_values)

# Print the result
print(pixel_count)



# Plot histogram
# plt.hist(pixel_values, bins=range(256), color='blue', alpha=0.7)
# plt.title('Pixel Value Distribution')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# %%
from PIL import Image

def get_image_channels(image_path):
    # Open the image
    image = Image.open(image_path)

    # Get the mode of the image
    mode = image.mode

    # Determine the number of channels
    if mode == 'L':
        return 1  # Grayscale image
    elif mode == 'RGB':
        return 3  # RGB image
    elif mode == 'RGBA':
        return 4  # RGBA image
    else:
        return f"Unknown mode: {mode}"

# Replace with the path to your image file
image_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amr_f06b03/benign/benign (1)_palette.png' #channel P
image_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amr_f06b03/benign/benign (1).png' #channel 1

channels = get_image_channels(image_path)
print(f"The image has {channels} channel(s).")

# %%
from PIL import Image
import numpy as np
# Replace 'path_to_image.jpg' with the path to your image file
img_path = '/home/lintzuh@kean.edu/BUS/AMR/result/ir_label_amrTzu_f045/benign/benign (1)_palette.png'

# Open the image file
img = Image.open(img_path)

# Convert the image to a numpy array
img_array = np.array(img)

# Print the shape of the image
print("Image shape:", img_array.shape)
# %%
