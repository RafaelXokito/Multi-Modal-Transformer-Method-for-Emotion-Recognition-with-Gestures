import cv2
import numpy as np

# Load the image
img = cv2.imread('22_Picnic_Picnic_22_688.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to the image to remove noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Canny Edge Detector to the image
edges = cv2.Canny(blurred, 50, 150)

# Display the result
cv2.imshow('Canny Edge Detector', edges)
cv2.imwrite("result/22_Picnic_Picnic_22_688_canny_edge_detector.jpg", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In this example, we first convert the image to grayscale as Canny Edge Detector works on grayscale images. Then, we apply Gaussian Blur to the image to remove noise. After that, we apply Canny Edge Detector to the image using the cv2.Canny function. The cv2.Canny function takes three arguments, the image, lower threshold, and upper threshold. Finally, we display the result using cv2.imshow function.