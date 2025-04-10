import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# denoised_image = cv2.medianBlur(image, 5)

# denoised_image = cv2.bilateralFilter(image,9,75,75)

denoised_image = cv2.fastNlMeansDenoising(image,None,30,7,21)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')

plt.show()