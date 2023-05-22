import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from IPython.display import display


path = '/Users/misaalsingh/Documents/Traffic Video Data/video'


def get_video(video):
    v_path = os.path.join(path, video)
    v_capture = cv2.VideoCapture(v_path)
    if not v_capture.isOpened():
        print('error reading the video')
    frames = []
    while v_capture.isOpened():
        ret, frame = v_capture.read()
        frames.append(frame)
        if not ret:
            break
    return frames

print(get_video('cctv052x2004080516x01638.avi'))

def view_video(video):
    v_path = os.path.join(path, video)
    v_capture = cv2.VideoCapture(v_path)
    if not v_capture.isOpened():
        print("Error opening video file")

    start_frame = 2
    v_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while v_capture.isOpened():
        ret, frame = v_capture.read()
        if not ret:
            print("End of video")
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    v_capture.release()
    cv2.destroyAllWindows()


def get_video_frame(video, f_num):
    v_path = os.path.join(path, video)
    v_capture = cv2.VideoCapture(v_path)
    if not v_capture.isOpened():
        print('error reading the video')
    frames = []
    while v_capture.isOpened():
        ret, frame = v_capture.read()
        frames.append(frame)
        if not ret:
            break

    image = frames[f_num]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    return image

def display_video(image_array):
    cv2.imshow('Image', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



