import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score


data2_normal = 'Data/ptbdb_normal.csv'
data2_anomaly = 'Data/ptbdb_abnormal.csv'

# read .csv
data2_normal = pd.read_csv(data2_normal, header=None).values
data2_anomaly = pd.read_csv(data2_anomaly, header=None).values

data2 = np.concatenate((data2_normal, data2_anomaly))
data = data2

X = data[:, 0:-1]
y = data[:, -1]

# 0 ist normal


# Train Test split of Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Normalization of Time-Series Data
class Scaler:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def minmax(self, scale_min=0.0, scale_max=1.0):
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

#scaler = Scaler(X_train, X_test)
#X_train, X_test = scaler.standardize()
#X_train, X_test = scaler.minmax()


y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

X_train_normal = X_train[~y_train]
X_test_normal = X_test[~y_test]

X_train_anomaly = X_train[y_train]
X_test_anomaly = X_test[y_test]


# Autoencoder Neural Network
model = tf.keras.Sequential([
    Dense(32, activation='relu', input_shape=(187,)),
    Dense(16, activation='relu'),
    Dense(12, activation='relu'),  # Bottleneck Layer
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(187, activation='relu')
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
                    epochs=150,
                    batch_size=500,
                    validation_data=(X_test, X_test),
                    shuffle=True)

# Plot Losses
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Training Reconstruction Loss
train_pred = model.predict(X_train_normal)
train_loss = tf.keras.losses.mae(train_pred, X_train_normal).numpy()
print('Train Loss', np.mean(train_loss))
threshold_loss = np.mean(train_loss) + 0.2 * np.std(train_loss)
sns.histplot(train_loss)
plt.show()

test_pred = model.predict(X_test)
test_loss = tf.keras.losses.mae(test_pred, X_test).numpy()
print('Test Loss', np.mean(test_loss))
sns.histplot(test_loss)
plt.show()


# Evaluation on unseen Test Dataset (Normal + Anomaly)
def eval_test_data(model, X, y, threshold):
    pred = model.predict(X)
    recon_loss = tf.keras.losses.mae(pred, X)
    pred_label = tf.math.less(recon_loss, threshold)
    print('Accuracy', accuracy_score(y, pred_label))
    print('Recall', recall_score(y, pred_label))
    print('Precision', precision_score(y, pred_label))

eval_test_data(model=model,
               X=X_test,
               y=y_test,
               threshold=threshold_loss)

def plot(X):
    a = model.predict(X)
    plt.plot(X)
    plt.plot(a)
    plt.show()