import cv2
import numpy as np

# Load the image
img = cv2.imread("pic.jpg")

# Define the diameter of the kernel for Bilateral Filter
diameter = 9

# Define the sigma color value for Bilateral Filter
sigma_color = 75

# Define the sigma space value for Bilateral Filter
sigma_space = 75

# Apply Bilateral Filter to the image
img_bilateral = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

# Display the original and filtered images
cv2.imshow("Original", img)
cv2.imshow("Bilateral Filter", img_bilateral)
cv2.imwrite("result/pic_bilateral.jpg", img_bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In this example, the bilateralFilter function takes four arguments:
# img: the input image.
# diameter: the diameter of the kernel used for the filter.
# sigma_color: the standard deviation for the color-space. Larger sigma_color values will result in stronger color smoothing.
# sigma_space: the standard deviation for the spatial-space. Larger sigma_space values will result in stronger spatial smoothing.