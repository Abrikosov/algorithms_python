import numpy as np

class PCA:
    """
    Principal component analysis (PCA)
    Linear dimensionality reduction by eigenvectors to a lower dimensional space.
    
    Parameters
    ----------
    n_components : int or None
        Number of components to keep.
        if n_components is not set all components are kept: n_components == min(n_samples, n_features)

    Attributes
    ----------
    components : array, shape (n_features, n_components)
        Principal axes. The components are sorted by aigenvalues`.
    mean_vector : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to "X.mean(axis=0)".
    n_components : int
        The estimated number of components. When n_components is set

    References
    ----------
    https://en.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), where n_samples in the number of samples
            and n_features is the number of features.
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape

        # Handle n_components==None
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        # Handle n_components > min(n_samples, n_features) or n_components < 1
        if not 1 <= self.n_components <= min(n_samples, n_features):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")

        # Center data
        self.mean_vector = np.mean(X, axis=0)
        X -= self.mean_vector

        # Covariance matrix
        self.cov_matrix = np.cov(X, rowvar=False)

        # Eigenvectors and aigenvalues from the covariance matrix
        eig_value, eig_vectors = np.linalg.eig(self.cov_matrix)
        eig_value_argsort_descending = np.abs(eig_value).argsort()[::-1]

        # Reduce eigenvectors
        self.components = eig_vectors[:, eig_value_argsort_descending[:self.n_components]]
        
        return X.dot(self.components)

if __name__ == "__main__":
    pass
 

