from sklearn.metrics.cluster import pair_confusion_matrix
from pca import PCA
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import normalize
from itertools import combinations

from model import Model
from kmeans import KMeans


# Load and transform the data
###############################################
# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# Reshape the data
x_trainr = x_train.reshape(-1, 1, 28, 28)
x_testr = x_test.reshape(-1, 1, 28, 28)
x_trainr = torch.from_numpy(x_trainr).float()
x_testr = torch.from_numpy(x_testr).float()

print("Training Samples dimension", x_trainr.shape)
print("Testing Samples dimension", x_testr.shape)

# Example of a picture
for i in range(9):
    ax = plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
################################################

# Train and laod the model
################################################
train = False  # Need to change to True if you want to train the model

type = 'small'  # Need to change to 'small', 'medium', or 'big' to train the model

if type == 'small':
    epochs = 10
elif type == 'medium':
    epochs = 100
elif type == 'big':
    epochs = 1000
else:
    epochs = 10

if train == True:
    # Train the model
    model = Model()
    model.train(x_trainr, batch_size=128, epochs=epochs)
    model.save('model_' + type + '.h5')
else:
    # Load the model
    model = Model()
    model.load('model_' + type + '.h5')

# Extract the features
x_test_encoded = model.extract_features(x_testr)
x_test_encoded = x_test_encoded.detach().numpy()

# Get the decoded ddata
n = 5
plt.figure(figsize=(20, 4))
decoded = model.forward(x_testr)
decoded = decoded.detach().numpy()

# Plot the decoded data
for i in range(n):
    # define subplot
    ax = plt.subplot(2, n, 1 + i)
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Real Data')
    ax = plt.subplot(2, n, 1 + i + n)
    plt.imshow(decoded[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Decoded Data')
plt.show()
################################################

# Clustering and evaluation
################################################
# Define the Adjusted Rand Index


def rand_index_score(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


# Fit KMeans model
kmeans = KMeans(n_clusters=10)
kmeans.fit(x_test_encoded)
labels = kmeans.predict(x_test_encoded)

# Evaluate the model
ari = rand_index_score(labels, y_test)
print(f'Adjusted Rand Index: {ari:.2f}')

# Reduceed the dimensionality of the data using PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_test_encoded)

# Plot the clustered data in 2D
plt.scatter(x_pca[:, 0], x_pca[:, 1],
            c=labels, cmap='viridis', alpha=0.7)
plt.title('K-means Clustering (k = 10) with PCA Visualization')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()
plt.show()
