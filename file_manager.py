import os
import pandas as pd
import numpy as np


path = '/Users/misaalsingh/Documents/Traffic Video Data/info.txt'
data_labels = []
with open(path, 'r') as file:
    rows = [line.strip() for line in file]
for row in rows:
    if 'light' in row:
        data_labels.append('light')
    elif 'medium' in row:
        data_labels.append('medium')
    elif 'heavy' in row:
        data_labels.append('heavy')

files = []
dir = '/Users/misaalsingh/Documents/Traffic Video Data/video'
for filename in os.listdir(dir):
    files.append(filename)
files = sorted(files)
del files[0]

data = dict(zip(files, data_labels))

ind = list(np.arange(len(files)))

data = pd.DataFrame({'Videos': files, 'Labels': data_labels}, index=ind)

print(ind)

