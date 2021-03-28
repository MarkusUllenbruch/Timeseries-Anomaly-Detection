import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Download the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
data = dataframe.values
print(dataframe.head())
print(dataframe.shape, data.shape)

# label and features
y = data[:, -1]
X = data[:, 0:-1]

# Train Test split of Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


# Normalization of Time-Series Data
class Scaler:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def minmax(self, scale_min=-1.0, scale_max=1.0):
        max_val = np.max(self.X_train)
        min_val = np.min(self.X_train)
        X_train = (scale_max - scale_min) * (self.X_train - min_val) / (max_val - min_val) + scale_min
        X_test = (scale_max - scale_min) * (self.X_test - min_val) / (max_val - min_val) + scale_min
        return X_train, X_test

    def standardize(self):
        mean = np.mean(self.X_train)
        std = np.std(self.X_train)
        X_train = (self.X_train - mean) / std
        X_test = (self.X_test - mean) / std
        return X_train, X_test

# Seperate Train and Test Data into Anomaly and Normal
scaler = Scaler(X_train, X_test)
X_train, X_test = scaler.minmax(scale_min=0.0, scale_max=1.0)
#X_train, X_test = scaler.standardize()


y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

X_train_normal = X_train[y_train]
X_test_normal = X_test[y_test]

X_train_anomaly = X_train[~ y_train]
X_test_anomaly = X_test[~ y_test]

# Autoencoder Neural Network
model = tf.keras.Sequential([
    Dense(32, activation='relu', input_shape=(140,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),  # Bottleneck Layer
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(140, activation='relu')
])
model.compile(optimizer='adam',
              loss='mae')
print(model.summary())

X_train_normal = tf.convert_to_tensor(X_train_normal, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)


# Train model on Normal Train Data (Normal Data Only)
# Validate Model on complete Test Set (Normal + Anomaly)
history = model.fit(X_train_normal,
                    X_train_normal,
                    epochs=110,
                    batch_size=1024,
                    validation_data=(X_test, X_test),
                    shuffle=True)

# Plot Losses
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Train Data Evaluation (Normal Only)
eps = 1.0  # Hyperparameter to Fine Tune Threshold Value (mean of Training Loss + 1 std)

train_reconstruction = model.predict(X_train_normal)
train_loss = tf.keras.losses.mae(train_reconstruction, X_train_normal)
threshold_loss = np.mean(train_loss) + eps * np.std(train_loss)
print('Threshold Reconstruction Loss', threshold_loss)

# Evaluation on unseen Test Dataset (Normal + Anomaly)
def eval_test_data(model, X, y, threshold):
    recon = model.predict(X)
    recon_loss = tf.keras.losses.mae(recon, X)
    pred_label =  tf.math.less(recon_loss, threshold)
    print('Accuracy', accuracy_score(y, pred_label))
    print('Recall', recall_score(y, pred_label))
    print('Precision', precision_score(y, pred_label))

eval_test_data(model=model,
               X=X_test,
               y=y_test,
               threshold=threshold_loss)
