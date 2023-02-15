import cv2

def non_local_means_filter(img, h=10, search_window=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h * 2, search_window)

# Load the input image
img = cv2.imread("pic.jpg")

# Apply the Non-local Means filter
img_filt = non_local_means_filter(img)

# Save the filtered image
cv2.imshow("Non-local Means Filter", img_filt)
cv2.imwrite("result/pic_non_local_means_filter.jpg", img_filt)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Note: In this code, the img argument should be a color image, and the h argument is a parameter that controls the strength of the denoising. The search_window argument is the size of the search window used by the filter.