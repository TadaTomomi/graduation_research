import cv2
import glob
import numpy as np
import torch

def make_data(data_directory):
    print("making data")
    datasets = []
    directory = sorted(glob.glob(data_directory))
    max = 100
    for data in directory:
        volume = []
        files = sorted(glob.glob(data + "/*.jpg"))
        num = int(np.ceil(len(files) / 3))
        for myFile in files[::3]:
            image1 = cv2.imread(myFile)
            ret, img_thresh = cv2.threshold(image1, 122, 255, cv2.THRESH_BINARY)
            #圧縮
            image2 = cv2.resize(img_thresh, (96, 96))
            volume.append(image2)
        zero = np.zeros(((max-num), 96, 96, 3))
        np_volume = np.array(volume)
        volume = np.append(np_volume, zero, axis=0).tolist()
        datasets.append(volume)
    torch_datasets = torch.tensor(datasets).permute(0, 4, 1, 2, 3)

    return torch_datasets.float()

# train_data = make_data("/home/student/datasets/CT/valid/*")
# print(train_data.shape)