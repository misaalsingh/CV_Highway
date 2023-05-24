import tensorflow as tf
from tensorflow import keras
from data_preprocessed import x_train, y_train, x_test, y_test
#define archetecture of the CNN

input_size = (53,240,320,3)

class CNN_3D(keras.Model):
    def __init__(self, input_size):
        super(CNN_3D, self).__init__()
        self.input_layer = keras.layers.InputLayer(input_shape=input_size)
        self.layer_1 = keras.layers.Conv3D(3, 3, activation='relu')
        self.pool_1 = keras.layers.MaxPool3D(pool_size=(3,3,3))
        self.layer_2 = keras.layers.Conv3D(3, 3, activation='relu')
        self.pool_2 = keras.layers.MaxPool3D(pool_size=(3,3,3))
        self.layer_3 = keras.layers.Conv3D(3, 3, activation='relu')
        self.pool_3 = keras.layers.MaxPool3D(pool_size=(3,3,3))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(99, activation='relu')
        self.fc2 = keras.layers.Dense(33, activation='relu')
        self.fc3 = keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
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
        return x

model = CNN_3D(input_size)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=tf.keras.metrics.Accuracy())
model.fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_test, y_test))
model.predict(x_test, y_test)
