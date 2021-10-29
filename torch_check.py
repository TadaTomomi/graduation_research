import torch

y = torch.tensor([[1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0]])
pred = torch.tensor([[1, 0, 3, 1, 0],
                    [0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0]])

# print("sex")
# print((y[:2].argmax() == pred[:2].argmax()).type(torch.float).sum().item())
# print("age")
# print((y[2:].argmax() == pred[2:].argmax()).type(torch.float).sum().item())


# print(y.argmax(1), pred.argmax(1))
# print((y.argmax(1) == pred.argmax(1)).type(torch.float).sum().item())

y_sex, y_age =  y.split([2, 3], 1)
pred_sex, pred_age = pred.split([2, 3], 1)
print("sex")
print((y_sex.argmax(1) == pred_sex.argmax(1)).type(torch.float).sum().item())
print("age")
print((y_age.argmax(1) == pred_age.argmax(1)).type(torch.float).sum().item())
