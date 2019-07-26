import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # Define and compile neural net.

model.compile(optimizer='sgd', loss='mean_squared_error')

# Providing data
xs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=int)
ys = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], dtype=int)

# Training neural net
model.fit(xs, ys, epochs=4000)
print(model.predict([50.0]))