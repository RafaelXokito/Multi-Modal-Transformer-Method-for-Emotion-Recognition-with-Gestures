import cv2
import numpy as np

# Load the input image
img = cv2.imread('images/pic_sobel.jpg')

# Create a 4-channel array from the input image
img_4ch = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
img_4ch[:, :, :3] = img

# Add the new fourth channel to the array
new_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 128
img_4ch[:, :, 3] = new_channel

# Save the 4-channel image
cv2.imwrite('output_image.jpg', img_4ch)