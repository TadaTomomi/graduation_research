import numpy as np
import torch

#Datasetクラスを作成
class CTDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_data = self.transform(self.data[idx])
        else:
          out_data = self.data[idx]
        out_label =  self.label[:, idx]  
        sex = torch.tensor(int(out_label[0]), dtype=torch.int64)
        age = torch.tensor(int(out_label[1]), dtype=torch.int64)  

        return out_data, torch.tensor([sex, age])