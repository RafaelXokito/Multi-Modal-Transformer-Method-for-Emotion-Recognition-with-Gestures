import cv2
import dlib
import numpy as np

# Load the input image
img = cv2.imread('../images/ffhq_621.png')

# Create the facial landmark detector using dlib's implementation
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Detect faces in the image using the detector
faces = detector(img, 1)

# For each face in the image, find the landmarks
for face in faces:
    landmarks = predictor(img, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Draw the landmarks on the image
    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

# Display the image with the landmarks marked
cv2.imshow("Facial Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()