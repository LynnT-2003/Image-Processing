# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:06:57 2023

@author: Lynn Thit Nyi Nyi
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the color image
img = cv.imread('rand.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Split the color image into individual channels
b, g, r = cv.split(img)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Apply CLAHE to each channel
cl1_b = clahe.apply(b)
cl1_g = clahe.apply(g)
cl1_r = clahe.apply(r)

# Merge the individual channels back into a color image
cl1 = cv.merge([cl1_b, cl1_g, cl1_r])

# Plot the original, CLAHE-applied images, and histograms side by side
plt.figure(figsize=(12, 6))

# Plot for the original image
plt.subplot(2, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

# Plot for the histogram of the original image
plt.subplot(2, 3, 4)
plt.hist(img.ravel(), 256, [0, 256], color='b')
plt.title('Histogram - Original Image')

# Plot for the CLAHE-applied image
plt.subplot(2, 3, 2)
plt.imshow(cv.cvtColor(cl1, cv.COLOR_BGR2RGB))
plt.title('CLAHE Applied Image')

# Plot for the histogram of the CLAHE-applied image
plt.subplot(2, 3, 5)
plt.hist(cl1.ravel(), 256, [0, 256], color='r')
plt.title('Histogram - CLAHE Applied Image')

plt.show()
