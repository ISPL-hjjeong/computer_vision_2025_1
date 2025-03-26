import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "./image/edgeDetectionImage.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Sobel filter to detect edges in x and y direction
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Calculate edge magnitude
magnitude = cv2.magnitude(sobel_x, sobel_y)

# Convert magnitude to uint8
edge_image = cv2.convertScaleAbs(magnitude)

# Display the original grayscale image and the edge image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Edge Strength Image")
plt.imshow(edge_image, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
