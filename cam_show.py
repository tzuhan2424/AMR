#%%
import numpy as np
import matplotlib.pyplot as plt
# Specify the path to the .npy file you want to load
import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_directory)
npy_file_path = './result/cams/benign/benign (4).npy'



# # Load the data from the .npy file
loaded_data = np.load(npy_file_path, allow_pickle=True).item()
# Assuming loaded_data contains your image data, replace this with the appropriate variable
image_data = loaded_data['high_res']

# Display the image using Matplotlib
plt.imshow(image_data[0])  # Use 'cmap' to specify the color map if needed
plt.axis('off')  # Turn off axis labels
plt.savefig('plot.png')

plt.show()
