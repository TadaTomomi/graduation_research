import cv2
import glob
import numpy as np
import torch

# def make_data(data_directory):
#     print("making data")
#     datasets = []
#     directory = sorted(glob.glob(data_directory))
#     max = 100
#     for data in directory:
#         volume = []
#         files = sorted(glob.glob(data + "/*.jpg"))
#         num = int(np.ceil(len(files) / 3))
#         for myFile in files[::3]:
#             image1 = cv2.imread(myFile)
#             #圧縮
#             image2 = cv2.resize(img1, (96, 96))
#             volume.append(image2)
#         zero = np.zeros(((max-num), 96, 96, 3))
#         np_volume = np.array(volume)
#         volume = np.append(np_volume, zero, axis=0).tolist()
#         datasets.append(volume)
#     torch_datasets = torch.tensor(datasets).permute(0, 4, 1, 2, 3)

#     return torch_datasets.float()

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
            image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            #圧縮
            image3 = cv2.resize(image2, (96, 96))
            image4 = np.array([image3])
            volume.append(image4)
        zero = np.zeros(((max-num), 1, 96, 96))
        np_volume = np.array(volume)
        volume = np.append(np_volume, zero, axis=0).tolist()
        datasets.append(volume)
    torch_datasets = torch.tensor(datasets).permute(0, 2, 1, 3, 4)

    return torch_datasets.float()

# train_data = make_data("/home/student/datasets/CT200/train/*")
# print(train_data.shape)