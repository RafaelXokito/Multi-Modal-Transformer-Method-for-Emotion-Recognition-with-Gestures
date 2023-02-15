import cv2
import os
import numpy as np

# Load the image dataset
dataset_folder = "../datasets/AffectNet_10Percent_div"
classes = os.listdir(dataset_folder)

filtered_dataset_folder = "../datasets/AffectNet_10Percent_div_Sobel"

# Define the filter function
def filter_image(img):

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel Filter to the grayscale image
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the gradient images in the x and y direction to obtain the final image
    img_sobel = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))
    filtered_img = (img_sobel * 255 / np.max(img_sobel)).astype(np.uint8)
    
    return filtered_img

# Apply the filter to each image in each class and division
for class_name in classes:
    class_folder = os.path.join(dataset_folder, class_name)
    divisions = os.listdir(class_folder)
    for division_name in divisions:
        division_folder = os.path.join(class_folder, division_name)
        images = []
        for filename in os.listdir(division_folder):
            img = cv2.imread(os.path.join(division_folder, filename))
            if img is not None:
                images.append(img)
        
        filtered_images = []
        for img in images:
            filtered_img = filter_image(img)
            filtered_images.append(filtered_img)
        
        # Save the filtered images
        filtered_division_folder = os.path.join(filtered_dataset_folder, class_name, division_name)
        if not os.path.exists(filtered_division_folder):
            os.makedirs(filtered_division_folder)
        for i, filtered_img in enumerate(filtered_images):
            cv2.imwrite(os.path.join(filtered_division_folder, "filtered_img_{}.jpg".format(i)), filtered_img)
