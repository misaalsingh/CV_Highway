import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape, StringLookup
import tensorflow.keras as keras

dir_path = '/Users/misaalsingh/Documents/tensorflow-test1/info.txt'
IMG_SIZE = 240
MAX_SEQ_LEN = 52
NUM_FEATURES = 2048
EPOCHS = 10

def get_file_names(path):
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
    dir = '/Users/misaalsingh/Documents/tensorflow-test1/video'
    for filename in os.listdir(dir):
        files.append(filename)
    files = sorted(files)
    del files[0]
    # Ensure the lengths match
    min_length = min(len(files), len(data_labels))
    files = files[:min_length]
    data_labels = data_labels[:min_length]

    data = dict(zip(files, data_labels))
    ind = list(np.arange(len(files)))
    data = pd.DataFrame({'Videos': files, 'Labels': data_labels}, index=ind)
    print(data.head())
    return data


VID_PATH = '/Users/misaalsingh/Documents/tensorflow-test1/video'
df = get_file_names(dir_path)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def pad_sequence(sequence, max_seq, padding_values=0):
    if len(sequence) < max_seq:
        pad_width  = ((0, max_seq - len(sequence)), (0,0), (0,0), (0,0))
        padded_sequence = np.pad(sequence, pad_width, mode='constant', constant_values=padding_values)
    else: 
        padded_sequence = sequence[:max_seq]
    return padded_sequence

def load_video(video, max_seq=52, padding_values=0):
    v_path = os.path.join(VID_PATH, video)
    v_capture = cv2.VideoCapture(v_path)
    v_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
    frames = []
    count = 0
    while True:
        ret, frame = v_capture.read()
        if not ret:
            break
        frame = frame / 255.0
        frame = crop_center_square(frame)
        frames.append(frame)
        count += 1
    v_capture.release()
    frames = np.array(frames)
    frames = pad_sequence(frames, 52)
    return frames

def view_video(video):
    v_path = os.path.join(VID_PATH, video)
    v_capture = cv2.VideoCapture(v_path)
    start_frame = 2
    v_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if not v_capture.isOpened():
        print("Error opening video file")
    while v_capture.isOpened():
        ret, frame = v_capture.read()
        if not ret:
            print("End of video")
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    v_capture.release()
    cv2.destroyAllWindows()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(df["Labels"])
)
print(label_processor.get_vocabulary())

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["Videos"].to_list()
    labels = df["Labels"].to_list()
    labels = tf.keras.utils.to_categorical(label_processor(labels), num_classes=len(label_processor.get_vocabulary()))
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LEN), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LEN, NUM_FEATURES), dtype="float32")

    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LEN), dtype="bool")
        temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LEN, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LEN, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)
            temp_frame_mask[i, :length] = 1

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


def train_test_split(df, test_size=0.25):
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df_shuffled) * 0.8)  # 80% for training, 20% for testing
    train_df = df_shuffled[:split_index]
    test_df = df_shuffled[split_index:]
    return train_df, test_df

train_df, test_df = train_test_split(df)

train_x, train_y = prepare_all_videos(train_df, VID_PATH)
test_x, test_y = prepare_all_videos(test_df, VID_PATH)

def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LEN, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LEN,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using mask:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model



def run_experiment():
    filepath = "/Users/misaalsingh/Documents/tensorflow-test1/ckpt.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_x[0], train_x[1]],
        train_y,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    return history, seq_model


_, sequence_model = run_experiment()
