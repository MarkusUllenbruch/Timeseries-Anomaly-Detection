# Anomaly-Detection & Classification of Time-Series Data
## Using FC, LSTM and CNN1D Autoencoder Neural Networks


This is my latest project in which I want to analyze univariate time series data with autoencoder neural networks. I want to implement:
- Unsupervised/ Semi Supervised Learning
- Classification
of time series data using an Autoencoder (AE) and Variational Autoencoder (VAE) Neural Networks.
I will test
- Fully Connected (FC)
- Long-Short-Term-Memory (LSTM)
- 1D Convolutional (CNN1D)
Neural Networks based on their performance on different metrics like Accuracy, Precision and Recall.

## Autoencoder
Autoencoder Neural Networks, in general, crunches the high-dimensional data into compressed, low-dimensional representation and then classifies anomalys based on the reconstruction-error of trying to reconstruct the original data from the compressed and low-dimensional data representation. Autoencoders are therefore a more general and nonlinear form of the linear Principal Component Analysis (PCA).

<img src="https://lilianweng.github.io/lil-log/assets/images/autoencoder-architecture.png" width="600">

## The Data
The used dataset is the ECG5000, which contains biosignals from humans. The data is a collection of heartbeats from an electrocardiogram.
Based on the assumption we will have many healthy samples, but very few abnormal samples, a semi supervised anomaly detection system will be implemented based on AE and VAE neural network structures.
