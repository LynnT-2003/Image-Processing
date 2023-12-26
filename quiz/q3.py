# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:29:31 2023

@author: Lynn Thit Nyi Nyi
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read color image
img_color = cv.imread('acm.jpg', cv.IMREAD_COLOR)

# Display the original color image
plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
plt.title('Original Color Image')
plt.show()

# Convert color image to RGB
img_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

# Extract R, G, and B components
r_channel = img_rgb[:, :, 0]
b_channel = img_rgb[:, :, 2]

# Perform histogram matching on R and B components
r_equ = cv.equalizeHist(r_channel)
b_equ = cv.equalizeHist(b_channel)

# Create new color image with equalized R and B components
img_equalized = img_rgb.copy()
img_equalized[:, :, 0] = r_equ
img_equalized[:, :, 2] = b_equ

# Display the result after histogram matching
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Color Image')

plt.subplot(1, 2, 2)
plt.imshow(img_equalized)
plt.title('After Histogram Matching')

plt.show()

# Plot histograms of R and B components
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(r_channel.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
plt.title('(Before) Histogram of R Component for whole image')

plt.subplot(1, 2, 2)
plt.hist(b_channel.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('(After) Histogram of B Component for whole image')

plt.show()

# Plot histograms of R and B components
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(r_equ.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
plt.title('(Before) Histogram of R Component for whole image')

plt.subplot(1, 2, 2)
plt.hist(b_equ.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('(After) Histogram of B Component for whole image')

plt.show()

# Perform local histogram equalization on the third quadrant of the color image
height, width, _ = img_rgb.shape
third_quadrant = img_rgb[height//2:, :width//2]

r_third_quadrant = third_quadrant[:, :, 0]
b_third_quadrant = third_quadrant[:, :, 2]

r_equ_third_quadrant = cv.equalizeHist(r_third_quadrant)
b_equ_third_quadrant = cv.equalizeHist(b_third_quadrant)

img_equalized_third_quadrant = third_quadrant.copy()
img_equalized_third_quadrant[:, :, 0] = r_equ_third_quadrant
img_equalized_third_quadrant[:, :, 2] = b_equ_third_quadrant

# Compute and plot the histogram of the color image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(r_third_quadrant.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
plt.title('Histogram of R Component in Third Quadrant')

plt.subplot(1, 2, 2)
plt.hist(b_third_quadrant.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('Histogram of B Component in Third Quadrant')

plt.show()

# Compute and plot the histogram of the color image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(r_equ_third_quadrant.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
plt.title('(Before) Histogram of R Component in 3rd Quadrant')

plt.subplot(1, 2, 2)
plt.hist(b_equ_third_quadrant.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('(After) Histogram of B Component in 3rd Quadrant')

plt.show()


# Display the result before and after local histogram equalization in the third quadrant
plt.figure(figsize=(12, 6))

# Before local histogram equalization
plt.subplot(1, 2, 1)
plt.imshow(third_quadrant)
plt.title('Before Local Histogram Equalization')


# After local histogram equalization
plt.subplot(1, 2, 2)
plt.imshow(img_equalized_third_quadrant)
plt.title('After Local Histogram Equalization')

plt.show()

