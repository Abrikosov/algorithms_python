import sys, os
import numpy as np

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from algorithms import dimension_reduction

EPSILON = 1e-3

class TestPCA:
    def test_fit_transform_can_restore(self):
        X = np.array([[-1., -1.], [-2., -1.], [-3., -2.], [1., 1.], [2., 1.], [3., 2.]])
        pca = dimension_reduction.PCA()
        X_new = pca.fit_transform(X)
        X_restored = X_new.dot(pca.components.T) + pca.mean_vector
        assert max(np.abs(X_restored - X)) <= EPSILON
        
    def test_fit_transform_from_2_to_1(self):
        X = np.array([[-1., -1.], [-2., -1.], [-3., -2.], [1., 1.], [2., 1.], [3., 2.]])
        X_true = np.array([[-1.38340578], [-2.22189802], [-3.6053038 ], [ 1.38340578], [ 2.22189802], [ 3.6053038 ]])
        pca = dimension_reduction.PCA(n_components=1)
        X_new = pca.fit_transform(X)
        assert np.max(np.abs(X_new - X_true)) <= EPSILON
