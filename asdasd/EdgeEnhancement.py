import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(image,cv2.CV_64F)
enhanced_image = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title('Edge Enhanced Image')
plt.imshow(enhanced_image, cmap='gray')

plt.show()