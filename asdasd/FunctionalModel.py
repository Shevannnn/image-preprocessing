import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

image_path = 'C:\\Users\\21-0270c\\Downloads\\img.jpg'
image = Image.open(image_path).convert('L')
image_array = np.array(image)

height, width = image_array.shape

x = np.arange(width)
y = np.arange(height)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = image_array.flatten()

poly = PolynomialFeatures(degree=5)

X = np.vstack((x, y)).T
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, z)

z_pred = model.predict(X_poly)
image_pred = z_pred.reshape(height, width)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image_array, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(image_pred, cmap='gray')
axs[1].set_title('Original Image')
axs[1].axis('off')

plt.show()

# This WHOLE logic is often just part of an object detection algorithm sequence.