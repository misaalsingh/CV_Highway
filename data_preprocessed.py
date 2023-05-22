import numpy as np
from sklearn.model_selection import train_test_split
from file_manager import data
import Video_Reader as vr
import tensorflow as tf
import pandas as pd
#making training and test sets

train_path = '/Users/misaalsingh/Documents/Traffic Video Data/Train.txt'
test_path = '/Users/misaalsingh/Documents/Traffic Video Data/Test.txt'
x = []

with open(train_path, 'r') as file:
    training_set = [line.strip() for line in file][0]
    training_set = training_set.split(',')

with open(test_path, 'r') as file:
    test_set = [line.strip() for line in file][0]
    test_set = test_set.split(',')

for i in range(len(training_set)):
    training_set[i] = int(training_set[i])

for i in range(len(test_set)):
    test_set[i] = int(test_set[i])

for i in data['Videos'].to_list():
    vid = vr.get_video(i)
    x.append(vid)

x = np.array(x, dtype=object)
y = np.array(data['Labels'])

x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=.3, random_state=42)

print(x_train.shape)
