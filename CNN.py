import tensorflow as tf
from tensorflow import keras
from file_manager import data
import Video_Reader as vr
import data_preprocessed as dr
#define archetecture of the CNN

input_size = (240,320,3)

class CNN_3D(keras.Model):
    def __init__(self, input_size):
        super(CNN_3D, self).__init__()
        self.input_size = input_size
        self.input = keras.layers.Input(input_size)
        self.layer_1 = keras.layers.Conv3D(3, 3, activation='relu', activitiy_regularizer='L2')
        self.pool_1 = keras.layers.MaxPool3D(pool_size=(3,3,3))
        self.layer_2 = keras.layers.Conv3D(3, 3, activation='relu', activitiy_regularizer='L2')
        self.pool_2 = keras.layers.MaxPool3D(pool_size=(3,3,3))
        self.layer_3 = keras.layers.Conv3D(3, 3, activation='relu', activitiy_regularizer='L2')
        self.pool_3 = keras.layers.MaxPool3D(pool_size=(3,3,3))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(99, activation='relu')
        self.fc2 = keras.layers.Dense(33, activation='relu')
        self.fc3 = keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.input(inputs)
        x = self.layer_1(x)
        x = self.pool_1(x)
        x = self.layer_2(x)
        x = self.pool_2(x)
        x = self.layer_3(x)
        x = self.pool_3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

model = CNN_3D()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=.01))
model.fit(dr.x_train, dr.y_train, epochs=20, batch_size=30, validation_data=(dr.x_test,dr.y_test))
