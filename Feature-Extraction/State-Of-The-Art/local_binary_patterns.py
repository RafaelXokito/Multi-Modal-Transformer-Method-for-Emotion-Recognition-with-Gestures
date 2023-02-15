import cv2
import numpy as np

# Load the input image
img = cv2.imread('../images/ffhq_621.png')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set up the LBP face detector
lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# Detect faces in the grayscale image using LBP
faces = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the faces marked
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()