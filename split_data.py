import os
from os.path import join as pjoin
import random

train_list = []
test_list = []
val_list = []

train_prob = 0.8
test_prob = 0.1
val_prob = 0.1
random.seed(3)

fd_train = open('data/train.txt', 'w')
fd_test = open('data/test.txt', 'w')
fd_val = open('data/val.txt', 'w')

file_list = os.listdir('data/top1000_complete')
for file in file_list:
    num = random.random()
    if num < train_prob:
        train_list.append(file)
        fd_train.write(file + '\n')
    elif num < train_prob + test_prob:
        test_list.append(file)
        fd_test.write(file + '\n')
    else:
        val_list.append(file)
        fd_val.write(file + '\n')

print(len(train_list), len(test_list), len(val_list))

fd_train.close()
fd_test.close()
fd_val.close()
