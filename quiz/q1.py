# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:24:27 2023

@author: Lynn Thit Nyi Nyi
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read color image
img_color = cv.imread('acm.jpg', cv.IMREAD_COLOR)

# Create a 1x3 subplot layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Display the original color image
axs[0].imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Color Image')

# Convert color image to RGB
img_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

# Extract R, G, and B components
r_channel = img_rgb[:, :, 0]
g_channel = img_rgb[:, :, 1]
b_channel = img_rgb[:, :, 2]

# Subtract distinct intensities from the maximum pixel intensity
max_intensity = 255

modified_image = np.zeros_like(img_rgb, dtype=np.uint8)
modified_image[:, :, 0] = max_intensity - r_channel
modified_image[:, :, 1] = max_intensity - g_channel
modified_image[:, :, 2] = max_intensity - b_channel

# Display the obtained image
axs[1].imshow(modified_image)
axs[1].set_title('Modified Image')

# Compute and display the difference between the original and modified images
difference_image = cv.absdiff(img_rgb, modified_image)

axs[2].imshow(difference_image)
axs[2].set_title('Difference Image')

# Show the combined plot
plt.show()
