import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, threshold1=80, threshold2=160)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(edges, cmap='gray')
axs[1].set_title('Edge Detection (Canny)')
axs[1].axis('off')

plt.show()