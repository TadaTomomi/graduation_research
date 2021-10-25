import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from make_data import train_data, valid_data
from make_label import make_label
from datasets import CTDataset
from model import CNN3D
from train import train
from valid import valid
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

#3次元CT(3次元配列)をnumpy配列に入れる
# train_data = torch.randn(5, 3, 216, 96, 96)
# valid_data = torch.randn(5, 3, 216, 96, 96)

train_data = torch.from_numpy(train_data()).float()
valid_data = torch.from_numpy(valid_data()).float()

#ラベルをnumpy配列に入れる
# train_label = torch.from_numpy(np.loadtxt('train_label.csv')).long()
# valid_label = torch.from_numpy(np.loadtxt('valid_label.csv')).long()

train_label = make_label('train_label.csv').float()
valid_label = make_label('valid_label.csv').float()

#transform(前処理を定義)
transform = transforms.Compose([
    transforms.ToTensor()
    ])

#データセットを定義
train_dataset = CTDataset(train_data, train_label, transform=None)
valid_dataset = CTDataset(valid_data, valid_label, transform=None)

#データローダーを定義
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

#モデルを読み込む
model = CNN3D(10).to(device)

print(model)

#損失関数
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

#最適化手法
optimizer = optim.Adam(model.parameters(), lr=0.01)

#学習
epochs = 10
valid_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, model, criterion, optimizer, device)
    valid_loss = valid(valid_dataloader, model, criterion, device)
    valid_loss_list.append(valid_loss)
print("Done!")

#モデルの保存
torch.save(model, 'model.pth')

#損失・正答率のグラフ表示
# plt.plot(train_loss, color='red')
plt.plot(valid_loss_list, color='blue')
plt.show()