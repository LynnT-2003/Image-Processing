# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:02:35 2023

@author: Lynn Thit Nyi Nyi
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def find_value_target(val, target_arr):
    key = np.where(target_arr == val)[0]

    if key.size == 0:
        return val  # Return the original value if not found in the target array

    vvv = key[0]
    return vvv

def match_histogram(inp_img, hist_input, e_hist_input, e_hist_target, _print=True):
    '''map from e_inp_hist to 'target_hist '''
    en_img = np.zeros_like(inp_img)
    tran_hist = np.zeros_like(e_hist_input)
    for i in range(len(e_hist_input)):
        tran_hist[i] = find_value_target(val=e_hist_input[i], target_arr=e_hist_target)
    '''enhance image as well:'''
    for x_pixel in range(inp_img.shape[0]):
        for y_pixel in range(inp_img.shape[1]):
            pixel_val = int(inp_img[x_pixel, y_pixel])
            en_img[x_pixel, y_pixel] = tran_hist[pixel_val]

    '''creating new histogram'''
    hist_img, _ = generate_histogram(en_img, print=False, index=3)
    return en_img, hist_img

def generate_histogram(img, print=True, index=1):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    if print:
        plt.subplot(1, 3, index)
        plt.plot(hist, color='black')
        plt.title(f'Histogram {index}')
        plt.xlim([0, 256])
        plt.show()

    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return hist, cdf_normalized

def print_img(img, histo_new, histo_old, index, L):
    '''Print final image'''
    plt.subplot(2, 3, index)
    plt.imshow(img, cmap='gray')  # Display in grayscale
    plt.title(f'Enhanced Image {index}')
    plt.subplot(2, 3, index + 3)
    plt.plot(histo_new, color='black')
    plt.plot(histo_old, color='red', alpha=0.5)
    plt.title(f'Histogram {index} Comparison')
    plt.xlim([0, L])
    plt.show()

# Read color image
img_color = cv.imread('acm.jpg', cv.IMREAD_COLOR)

# Create a 1x4 subplot layout
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Display the original color image
axs[0].imshow(cv.cvtColor(img_color, cv.COLOR_BGR2GRAY), cmap='gray')  # Convert to grayscale
axs[0].set_title('Original Grayscale Image')

# Convert color image to RGB
img_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

# Extract R and B components
r_channel = img_rgb[:, :, 0]
b_channel = img_rgb[:, :, 2]

# Compute and display histograms of the R and B components
hist_r, _ = generate_histogram(r_channel, print=False, index=1)
hist_b, _ = generate_histogram(b_channel, print=False, index=2)

axs[1].plot(hist_r, color='red')
axs[1].set_title('Histogram of R Component')
axs[2].plot(hist_b, color='blue')
axs[2].set_title('Histogram of B Component')

# Perform histogram matching on R and B components
r_equ, hist_r_equ = match_histogram(r_channel, hist_r, hist_r, hist_b)
b_equ, hist_b_equ = match_histogram(b_channel, hist_b, hist_b, hist_r)

# Create a new color image with matched R and B components
img_matched = img_rgb.copy()
img_matched[:, :, 0] = r_equ
img_matched[:, :, 2] = b_equ

# Display the result after histogram matching
axs[3].imshow(cv.cvtColor(img_matched, cv.COLOR_RGB2GRAY), cmap='gray')  # Convert to grayscale
axs[3].set_title('Enhanced Grayscale Image')

# Show the combined plot
plt.show()