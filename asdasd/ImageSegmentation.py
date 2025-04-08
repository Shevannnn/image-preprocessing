import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_,segmented = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(segmented, cmap='gray')
axs[1].set_title('Segmented Image (Otsu\'s Thresholding)')
axs[1].axis('off')

plt.show()