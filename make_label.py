import torch

from train import train

def make_label(file):
    csv_int = [list(map(int,line.rstrip().split(","))) for line in open(file).readlines()]
    csv_torch = torch.tensor(csv_int).long()
    return csv_torch

# train_label = make_label('train_label.csv').float()
# print(train_label.shape)