import numpy as np
from sklearn.model_selection import train_test_split
from file_manager import data
import Video_Reader as vr
from keras.utils import to_categorical
import pandas as pd

#add arrays of all the videos to an array

# Read frames from the video and store them in the list
x = np.zeros((254, 53, 240, 320, 3))
vid_count = 0
for i in data['Videos'].to_list():
    video_array = vr.get_video(i)
    x[vid_count] = video_array
    vid_count += 1

# convert the list to am array

y = np.array(data['Labels'])
y = to_categorical(y, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)



