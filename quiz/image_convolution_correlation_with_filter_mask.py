# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:20:08 2024

@author: Lynn Thit Nyi Nyi
"""

# https://www.linkedin.com/pulse/image-processing-convolution-filters-calculation-gradients-yadav/

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image_path = "dog.jpg"  # Replace with the path to your grayscale image
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define a filter mask (3x3 kernel) ! this is an example mask
# there exists several predefined masks for different detection purposes
filter_mask = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

# Perform convolution
convolution_result = cv2.filter2D(gray_image, -1, filter_mask)

# Perform correlation
correlation_result = cv2.filter2D(gray_image, -1, filter_mask[::-1, ::-1])  # Using flipped filter for correlation

# Display the original image, convolution result, and correlation result
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')

plt.subplot(1, 3, 2)
plt.imshow(convolution_result, cmap='gray')
plt.title('Convolution Result')

plt.subplot(1, 3, 3)
plt.imshow(correlation_result, cmap='gray')
plt.title('Correlation Result')

plt.show()
