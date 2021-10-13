import cv2
import glob
import numpy as np

volume = []
files = sorted(glob.glob("./data/*.jpg"))
for myFile in files:
    image = cv2.imread(myFile)
    volume.append(image)

print(np.array(volume).shape)