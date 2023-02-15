from face_detector import YoloDetector
import numpy as np
from PIL import Image

model = YoloDetector(target_size=720,min_face=90, device="mps")
orgimg = np.array(Image.open('ffhq_42.png'))
bboxes,points = model.predict(orgimg)

print(bboxes, points)