# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:50:17 2024

@author: Lynn Thit Nyi Nyi
"""

import cv2
import matplotlib.pyplot as plt

# Load the color image
image_path = "acm.jpg"
color_image = cv2.imread(image_path)
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# Extract R and B components
r_channel = color_image[:, :, 0]
b_channel = color_image[:, :, 2]

# Compute histograms of R and B components
hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])

# Plot the histograms before matching
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_r, color='red')
plt.title('Histogram - Red Channel (Before Matching)')

plt.subplot(1, 2, 2)
plt.plot(hist_b, color='blue')
plt.title('Histogram - Blue Channel (Before Matching)')

plt.show()

# Perform histogram matching using R and B components
matched_r_channel = cv2.equalizeHist(r_channel)
matched_b_channel = cv2.equalizeHist(b_channel)

# Convert matched R and B channels to grayscale
# matched_r_gray = cv2.cvtColor(cv2.merge([matched_r_channel, matched_r_channel, matched_r_channel]), cv2.COLOR_RGB2GRAY)
# matched_b_gray = cv2.cvtColor(cv2.merge([matched_b_channel, matched_b_channel, matched_b_channel]), cv2.COLOR_RGB2GRAY)

# Plot the histograms after matching
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(matched_r_channel.flatten(), bins=256, color='red', range=[0, 256])
plt.title('Histogram - Matched R Channel (After Matching)')

plt.subplot(1, 2, 2)
plt.hist(matched_b_channel.flatten(), bins=256, color='blue', range=[0, 256])
plt.title('Histogram - Matched B Channel (After Matching)')

plt.show()

# Display the original and matched R images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(matched_r_channel, cmap='gray')
plt.title('Histogram Matched R Channel (Grayscale)')

plt.show()

# Display the original and matched B images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(color_image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(matched_b_channel, cmap='gray')
plt.title('Histogram Matched B Channel (Grayscale)')

plt.show()


# Convert the original image to grayscale
original_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Display the original grayscale image
plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(original_gray, cmap='gray')
# plt.title('Original Grayscale Image')

# Perform histogram matching between matched R and B channels in grayscale
matched_b_r_matched = cv2.equalizeHist(matched_b_channel, matched_r_channel)

# Display the matched B and matched R images after histogram matching
plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(original_gray, cmap='gray')
# plt.title('Original Grayscale Image')

# plt.subplot(1, 3, 2)
# plt.imshow(matched_b_gray, cmap='gray')
# plt.title('Histogram Matched B Channel (Grayscale)')

# plt.subplot(1, 3, 3)
# plt.imshow(matched_r_gray, cmap='gray')
# plt.title('Histogram Matched R Channel (Grayscale)')

plt.show()

# Display the result of histogram matching between matched R and B channels in grayscale
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_gray, cmap='gray')
plt.title('Original Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(matched_b_r_matched, cmap='gray')
plt.title('Histogram Matched B Channel vs. R Channel (Grayscale)')

plt.show()