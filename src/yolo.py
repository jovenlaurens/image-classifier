import torch

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import normalize
from torchvision import transforms

from autoencoder import Autoencoder
from kmeans import KMeans

if __name__ == '__main__':
    # Load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalization
    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(20),
        transforms.GaussianBlur(5),
        AddGaussianNoise(0., 1.),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Apply the transform to your data
    x_trainr_transformed = torch.stack(x_train).float()
    x_trainr_transformed = [transform(x) for x in x_trainr_transformed]
    x_testr_transformed = torch.stack(x_test).float()
    x_testr_transformed = [transform(x) for x in x_testr_transformed]

    print(x_trainr_transformed.shape)
    # Reshape the data
    x_trainr = x_train.reshape(-1, 1, 28, 28)
    x_testr = x_test.reshape(-1, 1, 28, 28)
    x_trainr_transformed = x_trainr_transformed.reshape(-1, 1, 28, 28)
    x_testr_transformed = x_testr_transformed.reshape(-1, 1, 28, 28)

    print("Training Samples dimension", x_trainr.shape)
    print("Testing Samples dimension", x_testr.shape)

    x_trainr = torch.from_numpy(x_trainr).float()
    x_testr = torch.from_numpy(x_testr).float()

    from matplotlib import pyplot as plt

    for i in range(9):
        # define subplot
        ax = plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # show the figure
    plt.show()

    train = True

    type = 'small'

    if type == 'small':
        epochs = 10
    elif type == 'medium':
        epochs = 100
    elif type == 'big':
        epochs = 1000
    else:
        epochs = 10

    print('model_' + type + '.h5')
    # epoch_size = "small", epoch size = 10
    # epoch_size = "medium", epoch size = 100
    # epoch_size = "big", epoch size = 1000

    if train == True:
        model = Autoencoder()
        model.train(x_trainr_transformed, x_trainr,
                    batch_size=128, epochs=epochs)
        model.save('model_' + type + '.h5')
    else:
        model = Autoencoder()
        model.load('model_' + type + '.h5')

    x_test_encoded = model.extract_features(x_testr_transformed)

    x_test_encoded = x_test_encoded.detach().numpy()

    model.summary()

    n = 5

    plt.figure(figsize=(20, 4))

    decoded = model.forward(x_testr_transformed)

    decoded = decoded.detach().numpy()

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

    from kmeans import KMeans
    from pca import PCA
    from sklearn.metrics import adjusted_rand_score

    # Fit KMeans model
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(x_test_encoded)
    labels = kmeans.predict(x_test_encoded)
    ari = adjusted_rand_score(y_test, labels)

    print(f'Adjusted Rand Index: {ari:.2f}')

    # Plot the clustered data in 2D
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_test_encoded)
    plt.scatter(x_pca[:, 0], x_pca[:, 1],
                c=labels, cmap='viridis', alpha=0.7)
    plt.title('K-means Clustering (k = 10) with PCA Visualization')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar()
    plt.show()
