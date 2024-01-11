import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load grayscale image
image_path = 'acm.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Fourier transform
f_transform = np.fft.fft2(img)
f_transform_shifted = np.fft.fftshift(f_transform)

# Compute magnitude spectrum for visualization
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Display the original image and its magnitude spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()

# Modify the spectrum (e.g., remove high frequencies)
# For demonstration, let's remove a square region in the center
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
f_transform_shifted[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

# Apply inverse Fourier transform
f_transform_inverse = np.fft.ifftshift(f_transform_shifted)
img_filtered = np.fft.ifft2(f_transform_inverse).real

# Display the modified magnitude spectrum and the filtered image
plt.subplot(121), plt.imshow(np.log(np.abs(f_transform_shifted)), cmap='gray')
plt.title('Modified Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()