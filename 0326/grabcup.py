import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "./image/edgeDetectionImage.jpg"
image = cv2.imread(image_path)

# Initialize mask, background model, and foreground model
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define the rectangle for the initial object selection
height, width = image.shape[:2]
rect = (int(width * 0.25), int(height * 0.25), int(width * 0.5), int(height * 0.5))  # (x, y, w, h)

# Apply GrabCut
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create mask where sure and probable foreground are set to 1, others to 0
mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
result = image * mask2[:, :, np.newaxis]  # Apply mask to original image

# Visualize results
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("GrabCut Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Foreground Extracted")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
