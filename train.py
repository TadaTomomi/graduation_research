import numpy as np
import torch
import torch.nn.functional as F

def train(dataloader, model, loss_fn1, loss_fn2, optimizer, device):
    size = len(dataloader.dataset)
    loss_sex_total = 0.0
    loss_age_total = 0.0
    epoch_loss = 0.0
    correct_sex, correct_age = 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred_sex, pred_age = model(X)
      
        #yをsexとageに分ける
        y_sex = y[:, 0]
        y_age = y[:, 1]

        loss_sex = loss_fn1(pred_sex, y_sex)
        loss_age = loss_fn2(pred_age, y_age)

        loss = loss_sex + loss_age
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sex_total += loss_sex.item()
        loss_age_total += loss_age.item()
        epoch_loss += (loss_sex.item() + loss_age.item())

        #正答率の計算

        correct_sex += (y_sex == pred_sex.argmax(1)).type(torch.float).sum().item()
        correct_age += (y_age == pred_age.argmax(1)).type(torch.float).sum().item()

    loss_sex_total /= size
    loss_age_total /= size
    epoch_loss /= size
    correct_sex /= size
    correct_age /= size
    correct_sex *= 100
    correct_age *= 100

    print("Train:")
    print(f"Avg loss_sex: {loss_sex_total:>8f}")
    print(f"Avg loss_age: {loss_age_total:>8f}")
    print(f"Avg loss_total: {epoch_loss:>8f} \n")
    print(f"Accuracy(sex): {(correct_sex):>0.1f}%")
    print(f"Accuracy(age): {(correct_age):>0.1f}% \n")
    
    return [epoch_loss, loss_sex_total, loss_age_total], correct_sex, correct_age