import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorize(im, color, clip_percentile=0.1):
    """
    Helper function to create an RGB image from a single-channel image using a 
    specific color.
    """
    # Check that we do just have a 2D image
    if im.ndim > 2 and im.shape[2] != 1:
        raise ValueError('This function expects a single-channel image!')
        
    # Rescale the image according to how we want to display it
    im_scaled = im.astype(np.float32) - np.percentile(im, clip_percentile)
    im_scaled = im_scaled / np.percentile(im_scaled, 100 - clip_percentile)
    im_scaled = np.clip(im_scaled, 0, 1)
    
    # Need to make sure we have a channels dimension for the multiplication to work
    im_scaled = np.atleast_3d(im_scaled)
    
    # Reshape the color (here, we assume channels last)
    color = np.asarray(color).reshape((1, 1, -1))
    return im_scaled * color

im = cv2.imread('output_image.png')

im_red = colorize(im[..., 1], (1, 0, 0))
plt.imshow(im_red)
plt.axis(False)
plt.title('Red')
plt.show()

im_green = colorize(im[..., 1], (0, 1, 0))
plt.imshow(im_green)
plt.axis(False)
plt.title('Green')
plt.show()

im_red = colorize(im[..., 1], (0, 0, 1))
plt.imshow(im_red)
plt.axis(False)
plt.title('Blue')
plt.show()

im_red = colorize(im[..., 1], (0, 0, 0, 1))
plt.imshow(im_red)
plt.axis(False)
plt.title('Landmarks')
plt.show()
