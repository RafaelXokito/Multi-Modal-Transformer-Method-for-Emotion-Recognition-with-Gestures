import dlib
import cv2
import numpy as np

# Load the input image
image = cv2.imread('../images/pic.jpg')

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Detect faces in the image
faces = detector(image)

# Loop over the detected faces and apply segmentation to each one
for face in faces:
    # Get the facial landmarks for the face
    landmarks = predictor(image, face)
    
    # Create a binary mask for the face using the landmarks
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 68)])], (255, 255, 255))
    
    # Apply the mask to the face to isolate it
    face_segmented = cv2.bitwise_and(image, mask)

    # Display the segmented face
    cv2.imshow('Segmented Face', face_segmented)
    cv2.waitKey(0)

cv2.destroyAllWindows()