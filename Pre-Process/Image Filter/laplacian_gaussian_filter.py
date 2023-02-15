import cv2
import numpy as np

# Load the image
img = cv2.imread('22_Picnic_Picnic_22_688.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to the image to remove noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Laplacian of Gaussian (LoG) filter to the image
log_filter = cv2.Laplacian(blurred, cv2.CV_64F)

# Normalize the result for display
log_filter = np.uint8(np.absolute(log_filter))

# Display the result
cv2.imshow('Laplacian of Gaussian (LoG) Filter', log_filter)
cv2.imwrite("result/22_Picnic_Picnic_22_688_laplacian_gaussian_filter.jpg", log_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In this example, we first convert the image to grayscale as LoG filter works on grayscale images. Then, we apply Gaussian Blur to the image to remove noise. After that, we apply Laplacian of Gaussian (LoG) filter to the image using the cv2.Laplacian function. Finally, we display the result using cv2.imshow function.