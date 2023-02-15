# import libraries
from yoloface import face_analysis
import numpy
import cv2
face=face_analysis()        #  Auto Download a large weight files from Google Drive.
                            #  only first time.
                            #  Automatically  create folder .yoloface on cwd.
# example 1
img,box,conf=face.face_detection(image_path='test.png',model='full')
print(box)                  # box[i]=[x,y,w,h]
print(conf)                 #  value between(0 - 1)  or probability
output_frame = face.show_output(img,box,frame_status=True)

cv2.imshow('image',img)
cv2.waitKey(0)