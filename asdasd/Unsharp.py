import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image,(9,9),10.0)
sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title('Sharpened Image')
plt.imshow(sharpened, cmap='gray')

plt.show()