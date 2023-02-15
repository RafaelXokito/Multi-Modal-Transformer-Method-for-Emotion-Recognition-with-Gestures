import cv2
import numpy as np

# Load the image
img = cv2.imread("22_Picnic_Picnic_22_688.jpg")

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Sobel Filter to the grayscale image
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

# Combine the gradient images in the x and y direction to obtain the final image
img_sobel = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))
img_sobel = (img_sobel * 255 / np.max(img_sobel)).astype(np.uint8)

# Display the original and filtered images
cv2.imshow("Original", img)
cv2.imshow("Sobel Filter", img_sobel)
cv2.imwrite("result/22_Picnic_Picnic_22_688_sobel.jpg", img_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In this example, the Sobel function takes five arguments:

# img_gray: the input grayscale image.
# cv2.CV_64F: the data type used for the output image.
# 1: the order of the derivative in the x direction.
# 0: the order of the derivative in the y direction.
# ksize: the size of the Sobel kernel used for the filter. A larger ksize value will result in a more pronounced edge detection.