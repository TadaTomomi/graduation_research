import cv2
import glob
import numpy as np

def train_data():
    datasets = []
    directory = glob.glob("/home/student/datasets/CT/train/*")
    num = len(directory)
    for data in directory:
        volume = []
        files = sorted(glob.glob(data + "/*.jpg"))
        for myFile in files:
            image1 = cv2.imread(myFile)
            #圧縮
            image2 = cv2.resize(image1, (96, 96))
            volume.append(image2)
        datasets.append(volume)
    np_datasets = np.array(datasets)
    np_datasets = np_datasets.reshape([num, 3, 216, 96, 96])
    return np_datasets

def valid_data():
    datasets = []
    directory = glob.glob("/home/student/datasets/CT/valid/*")
    num = len(directory)
    for data in directory:
        volume = []
        files = sorted(glob.glob(data + "/*.jpg"))
        for myFile in files:
            image1 = cv2.imread(myFile)
            #圧縮
            image2 = cv2.resize(image1, (96, 96))
            volume.append(image2)
        datasets.append(volume)
    np_datasets = np.array(datasets)
    np_datasets = np_datasets.reshape([num, 3, 216, 96, 96])
    return np_datasets