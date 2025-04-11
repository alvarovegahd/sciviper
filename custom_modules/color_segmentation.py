import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image (OpenCV loads in BGR format)
image = cv2.imread('custom_modules/images/Spectrogram-1.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV color space (suitable for color-based masking)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for high-energy regions (e.g., yellow to red hues)
# You can tweak the HSV values to best match your image
lower = np.array([10, 100, 100])   # Approx. starting from orange/yellow
upper = np.array([35, 255, 255])  # Ending at yellow/orange

# Create a binary mask
mask = cv2.inRange(image_hsv, lower, upper)

# Apply the mask (optional, creates segmented image)
segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Mask (High Energy Areas)')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Segmented Image')
plt.imshow(segmented)
plt.axis('off')

plt.tight_layout()
plt.show()
cv2.imwrite('custom_modules/images/segmented_image.png', segmented)  # Save segmentation result (if needed)

# Continue: assume mask already exists
import cv2
import numpy as np

# Find contours (cv2.RETR_EXTERNAL: only outer contours, cv2.CHAIN_APPROX_SIMPLE: compress contour representation)
all_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# For each contour, compute bounding box and area
contours = []
for i, cnt in enumerate(all_contours):
    area = cv2.contourArea(cnt)
    if area > 50:  # Filter out small regions considered as noise
        contours.append(cnt)

# Copy image for drawing contours
contour_only_img = image.copy()

# Draw the actual contour lines
cv2.drawContours(contour_only_img, contours, -1, (0, 255, 0), 2)

# Show with proper color space
plt.figure(figsize=(10, 6))
plt.title('High-Energy Region Outlines (Contours)')
plt.imshow(cv2.cvtColor(contour_only_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Optionally save it
cv2.imwrite('custom_modules/images/contours_outline.png', contour_only_img)


# Copy the original image (for visualizing bounding boxes)
contour_img = image.copy()

# For each contour, compute bounding box and area
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(contour_img, f'{i}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Visualize the contour result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.title('Detected High-Energy Regions (Contours)')
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))  # Fix color reversal
plt.axis('off')
plt.show()
cv2.imwrite('custom_modules/images/contour_image.png', contour_img)  # Save contour result (if needed)

# Print each region's info
print("Detected regions:")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 50:
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"[{i}] Bounding box: (x={x}, y={y}, w={w}, h={h}), Area: {area}")
