import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # Define and compile neural net.

model.compile(optimizer='sgd', loss='mean_squared_error')

# Providing data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=float)
ys = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0], dtype=float)

# Training neural net
model.fit(xs, ys, epochs=4000)
print(model.predict([50.0]))  # Will predict value for ys = 50.0
