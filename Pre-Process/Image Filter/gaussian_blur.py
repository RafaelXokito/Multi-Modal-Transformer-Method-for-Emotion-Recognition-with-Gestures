import cv2
import numpy as np

# Load the image
img = cv2.imread("22_Picnic_Picnic_22_688.jpg")

# Define the kernel size for Gaussian Blur
kernel_size = (15, 15)

# Apply Gaussian Blur to the image
img_blur = cv2.GaussianBlur(img, kernel_size, 0)

# Display the original and blurred images
cv2.imshow("Original", img)
cv2.imshow("Blurred", img_blur)
cv2.imwrite("result/22_Picnic_Picnic_22_688_gaussian_blur.jpg", img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

#In this example, the GaussianBlur function takes three arguments:
#img: the input image.
#kernel_size: the size of the kernel used for blurring. A larger kernel size will result in a more intense blur effect.
#0: the standard deviation of the Gaussian distribution used for blurring. A value of 0 means that the standard deviation will be calculated automatically based on the kernel size.