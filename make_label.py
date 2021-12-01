import csv
import torch
import torch.nn.functional as F

from train import train

def make_label(file):
    csv_int = [list(map(int,line.rstrip().split(","))) for line in open(file, encoding = "utf-8-sig").readlines()]
    csv_torch = torch.tensor(csv_int)
    return csv_torch.permute(1, 0)

# train_label = make_label('train_label.csv')
# print(train_label)
# print(train_label.shape)