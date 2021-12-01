import numpy as np
import torch

def valid(dataloader, model, loss_fn1, loss_fn2, device):
    size = len(dataloader.dataset)
    model.eval()
    epoch_loss = 0.0
    loss_sex, loss_age = 0.0, 0.0
    correct_sex, correct_age = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_sex, pred_age = model(X)
            
            #yをsexとageに分ける
            y_sex = y[:, 0]
            y_age = y[:, 1]
            
            loss_sex += loss_fn1(pred_sex, y_sex).item()
            loss_age += loss_fn2(pred_age, y_age).item()

            correct_sex += (y_sex == pred_sex.argmax(1)).type(torch.float).sum().item()
            correct_age += (y_age == pred_age.argmax(1)).type(torch.float).sum().item()

    loss_sex /= size
    loss_age /= size
    epoch_loss = loss_sex + loss_age
    correct_sex /= size
    correct_age /= size
    correct_sex *= 100
    correct_age *= 100
    
    print("Valid:")
    print(f"Avg loss_sex: {loss_sex:>8f}")
    print(f"Avg loss_age: {loss_age:>8f}")
    print(f"Avg loss_total: {epoch_loss:>8f} \n")
    print(f"Accuracy(sex): {(correct_sex):>0.1f}%")
    print(f"Accuracy(age): {(correct_age):>0.1f}% \n")

    return [epoch_loss, loss_sex, loss_age], correct_sex, correct_age