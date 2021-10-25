import numpy as np
import cv2
import glob
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('model.pth')
model.to(device)

datasets = []
volume = []
files = sorted(glob.glob("/home/student/datasets/CT/valid/01465240.31y8m.f/*.jpg"))
for myFile in files:
  image1 = cv2.imread(myFile)
  #圧縮
  image2 = cv2.resize(image1, (96, 96))
  volume.append(image2)
datasets.append(volume)
datasets = np.array(datasets).reshape((1, 3, 216, 96, 96))
test_data = torch.from_numpy(datasets).float()
# print(test_data.shape)

class_label = ['male', 'female']
test_data = test_data.to(device)
model.eval()
pred = model(test_data)
mx_id = torch.argmax(pred)
print(pred)
print(class_label[mx_id])