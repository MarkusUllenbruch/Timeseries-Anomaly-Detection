import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, InputLayer, Flatten, Reshape
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# Load Data
(X_train, _), (X_test, _) = fashion_mnist.load_data()

# Normalize Images to 0..1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Neural Network Class
class Autoencoder(Model):
    def __init__(self, bottleneck_dim=64):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        self.encoder = tf.keras.Sequential([
            InputLayer(input_shape=(28, 28)),
            Flatten(),
            Dense(units=self.bottleneck_dim, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            Dense(units=784, activation='relu'),
            Reshape(target_shape=(28, 28))
        ])

    def __call__(self, inputs, *args, **kwargs):
        encode = self.encoder(inputs)
        return self.decoder(encode)


autoencoder = Autoencoder(bottleneck_dim=64)
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

autoencoder.fit(
    x=X_train,
    y=X_train,
    epochs=10,
    shuffle=True,
    validation_data=(X_test, X_test))


test_preds = autoencoder.predict(X_test)
print(test_preds.shape, X_test.shape)


for num in range(10):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[num])
    plt.gray()
    plt.subplot(1, 2, 2)
    plt.imshow(test_preds[num])
    plt.gray()
    plt.show()






