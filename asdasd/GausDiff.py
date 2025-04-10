import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

blur1 = cv2.GaussianBlur(image,(5,5),1)
blur2 = cv2.GaussianBlur(image,(5,5),2)

dog = blur1 - blur2

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title('woof Image')
plt.imshow(dog, cmap='gray')

plt.show()
