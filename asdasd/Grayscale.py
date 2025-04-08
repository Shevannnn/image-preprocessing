from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:\\Users\\21-0270c\\Downloads\\img.jpg'
image = Image.open(image_path)

image_gray = image.convert('L')

pixel_grid = np.array(image_gray)

plt.imshow(pixel_grid, cmap='gray')
plt.title('Pixel Grid Representation')
plt.colorbar()
plt.show()

print("Pixel Grid Values:")
print(pixel_grid)