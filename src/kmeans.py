import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(
            X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)

            # Update centroids based on the mean of data points in each cluster
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return self

    def _assign_labels(self, X):
        # Compute distances from data points to centroids
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids, axis=2)

        # Assign each data point to the cluster with the nearest centroid
        labels = np.argmin(distances, axis=1)

        return labels

    def predict(self, X):
        # Assign data points to the nearest centroid
        return self._assign_labels(X)

    def get_centroids(self):
        # Return the centroids
        return self.centroids


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    X = np.random.rand(100, 2)  # Example 2D data with 100 samples
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.get_centroids()

    print("Cluster Labels:", labels)
    print("Centroids:", centroids)
