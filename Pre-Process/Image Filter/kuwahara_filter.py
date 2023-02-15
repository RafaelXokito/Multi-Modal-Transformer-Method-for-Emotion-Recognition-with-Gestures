import cv2
import numpy as np

def kuwahara_filter(img, kernel_size=15):
    h, w = img.shape[:2]
    padding = kernel_size // 2
    img_pad = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    img_filt = np.zeros((h, w), dtype=np.float32)
    for y in range(padding, h + padding):
        for x in range(padding, w + padding):
            region = img_pad[y - padding:y + padding + 1, x - padding:x + padding + 1]
            stats = [np.mean(region[0:padding, 0:padding]), np.mean(region[0:padding, padding:2 * padding + 1]),
                     np.mean(region[padding:2 * padding + 1, 0:padding]), np.mean(region[padding:2 * padding + 1, padding:2 * padding + 1])]
            index = np.argmin(stats)
            img_filt[y - padding, x - padding] = stats[index]
    return img_filt

# Load the input image
img = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE)


# Apply the Kuwahara filter
img_filt = kuwahara_filter(img)

# Normalize the result for display
img_filt = np.uint8(np.absolute(img_filt))

# Save the filtered image
cv2.imshow("Kuwahara Filter", img_filt)
cv2.imwrite("result/pic_kuwahara_filter.jpg", img_filt)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Note: In this code, the img argument should be a grayscale image, and the kernel_size argument should be an odd integer that specifies the size of the sliding window for the filter. The default value is 15.