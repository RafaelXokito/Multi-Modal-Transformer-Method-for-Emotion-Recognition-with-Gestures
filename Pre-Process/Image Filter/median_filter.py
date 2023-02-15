import cv2
import numpy as np

# Load the image
img = cv2.imread("pic.jpg")

# Define the kernel size for Median Filter
kernel_size = 5

# Apply Median Filter to the image
img_median = cv2.medianBlur(img, kernel_size)

# Display the original and filtered images
cv2.imshow("Original", img)
cv2.imshow("Median Filter", img_median)
cv2.imwrite("result/pic_median_filter.jpg", img_median)
cv2.waitKey(0)
cv2.destroyAllWindows()

#In this example, the medianBlur function takes three arguments:
#img: the input image.
#kernel_size: the size of the kernel used for the median filter. The kernel size should be an odd number. Larger kernel sizes will result in a stronger filter effect.