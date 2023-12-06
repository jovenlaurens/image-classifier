import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # Convert the list of vectors to a 2D NumPy array
        X_array = np.array(X)

        # Calculate the mean vector
        mean_vector = np.mean(X_array, axis=0)

        # Center the data by subtracting the mean
        centered_data = X_array - mean_vector

        # Calculate the covariance matrix
        cov_matrix = np.cov(centered_data, rowvar=False)

        # Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Choose the top n_components eigenvectors
        transform_matrix = sorted_eigenvectors[:, :self.n_components]

        # Project the original data onto the new space
        reduced_data = np.dot(centered_data, transform_matrix)

        return reduced_data


if __name__ == '__main__':
    X = [[1, 2, 3, 16], [4, 5, 6, 14], [7, 8, 9, 11]]  # List of vectors
    pca = PCA(n_components=2)  # Adjust n_components as needed
    reduced_data = pca.fit_transform(X)

    print("Original Data:")
    print(np.array(X))
    print("\nReduced Data:")
    print(reduced_data)
