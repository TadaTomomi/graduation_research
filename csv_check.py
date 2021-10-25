import numpy as np

# with open('gender.csv') as f:
#     print(f.read())

# import csv

# with open('check.csv') as fp:
#     lst = list(csv.reader(fp))

# print(lst)

csv_int = [list(map(int,line.rstrip().split(","))) for line in open("train_label.csv").readlines()]
print(csv_int)

# train_label = np.loadtxt('check.csv')
# print(train_label)
# print(train_label.shape)