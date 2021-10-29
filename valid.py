import numpy as np
import torch

# def valid(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     model.eval()
#     valid_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             valid_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     valid_loss /= size
#     correct /= size
#     print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")

#     return valid_loss

def valid(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    valid_loss, correct_sex, correct_age = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            y_sex, y_age =  y.split([2, 8], 1)
            pred_sex, pred_age = pred.split([2, 8], 1)
            correct_sex += (y_sex.argmax(1) == pred_sex.argmax(1)).type(torch.float).sum().item()
            correct_age += (y_age.argmax(1) == pred_age.argmax(1)).type(torch.float).sum().item()
    valid_loss /= size
    correct_sex /= size
    correct_age /= size
    print("Valid Error:")
    print(f"Accuracy(sex): {(100*correct_sex):>0.1f}%")
    print(f"Accuracy(age): {(100*correct_age):>0.1f}%")
    print(f"Avg loss: {valid_loss:>8f} \n")

    return valid_loss