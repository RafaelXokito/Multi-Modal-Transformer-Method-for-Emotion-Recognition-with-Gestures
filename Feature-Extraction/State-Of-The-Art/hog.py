import cv2
import numpy as np

# Load the input image
img = cv2.imread('../images/ffhq_621.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set the parameters for HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect objects in the grayscale image using HOG
detected_objs, _ = hog.detectMultiScale(gray_img, winStride=(8, 8), padding=(32, 32), scale=1.05)

# Draw rectangles around the detected objects
for (x, y, w, h) in detected_objs:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the objects marked
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
