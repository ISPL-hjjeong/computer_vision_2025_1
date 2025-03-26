import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reload the new image
image_path = "./image/edgeDetectionImage.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detector
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Apply Hough Line Transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Copy original image to draw lines
line_image = image.copy()

# Draw detected lines in red color
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display original + edge lines image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Canny Edge Map")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Detected Lines on Original")
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
