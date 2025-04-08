from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = 'C:\\Users\\21-0270c\\Downloads\\img.jpg'
image = Image.open(image_path)

image_rgb = image.convert('RGB')
pixel_grid_rgb = np.array(image_rgb)

image_hsv = image.convert('HSV')
pixel_grid_hsv = np.array(image_hsv)

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
pixel_grid_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)

fig, axs = plt.subplots(1, 3, figsize=(15,5))

axs[0].imshow(pixel_grid_rgb)
axs[0].set_title('RGB Color Space')
axs[0].axis('off')

axs[1].imshow(pixel_grid_hsv)
axs[1].set_title('HSV Color Space')
axs[1].axis('off')

axs[2].imshow(cv2.cvtColor(pixel_grid_lab, cv2.COLOR_LAB2RGB))
axs[2].set_title('LAB Color Space')
axs[2].axis('off')

plt.show()

print("Pixel Grid Values (RGB):")
print(pixel_grid_rgb)

print("\nPixel Grid Values (RGB):")
print(pixel_grid_hsv)

print("\nPixel Grid Values (RGB):")
print(pixel_grid_lab)
