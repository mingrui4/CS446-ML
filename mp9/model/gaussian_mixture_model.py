"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        # self._mu = np.random.normal(size=(n_components, n_dims))  # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        self._pi = [1./self._n_components] * self._n_components   # np.array of size (n_components, 1)

        # Initialized with identity.
        self._sigma = [np.eye(self._n_dims)] * self._n_components * 1000  # np.array of size (n_components, n_dims, n_dims)
    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        self._mu = x[np.random.choice(len(x), self._n_components, False), :]
        for i in range(self._max_iter):
            z_ik = self._e_step(x)
            self._m_step(x, z_ik)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = self.get_posterior(x)
        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        N_ks = np.sum(z_ik, axis=0)
        for k in range(self._n_components):
            ## means
            self._mu[k] = 1. / N_ks[k] * np.sum(z_ik[:, k] * x.T, axis=1).T
            x_mu = np.matrix(x - self._mu[k])

            ## covariances
            self._sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T, z_ik[:, k]), x_mu))

            ## and finally the probabilities
            self._pi[k] = 1. / len(x) * N_ks[k]


    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = np.zeros((len(x), self._n_components))
        # P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-x.shape[1] / 2.) \
        #                   * np.exp(-.5 * np.einsum('ij, ij -> i',x - mu,
        #                                            np.dot(np.linalg.inv(s), (x - mu).T).T))+self._reg_covar
        #
        # for k in range(self._n_components):
        #     ret[:, k] = self._pi[k] * P(self._mu[k], self._sigma[k])

        for k in range(self._n_components):
            ret[:,k] = self._pi[k]*multivariate_normal.pdf(x, self._mu[k], self._sigma[k])+ self._reg_covar

        return np.array(ret)

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        ret = self.get_conditional(x)
        sum_ret = np.sum(ret, axis=1)
        return sum_ret

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        ret = self.get_conditional(x)
        sum_ret = self.get_marginals(x)
        z_ik = (ret.T / sum_ret).T
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, 1)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        post = self.get_posterior(x)
        labels = np.argmax(post, axis=1)
        count = np.zeros((self._n_components,self._n_components))
        for cluster,label in zip(labels,y):
            count[cluster,int(label)] +=1
        self.cluster_label_map = np.argmax(count, axis =1)

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        z_ik = self.get_posterior(x)
        cluster = np.argmax(z_ik,axis=1)
        y_hat = [self.cluster_label_map[k] for k in cluster]
        return np.array(y_hat)
