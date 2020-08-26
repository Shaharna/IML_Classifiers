import numpy as np

class QDA:

    def __init__(self):
        self.samples_train = None
        self.labels_train = None
        self.mu = []
        self.phi = []
        self.sigma_matrix = []
        self.x_labels = []
        self.eigen = []


    def fit(self, x, y):
        """
        This function builds the QDA parameters according to the
         train values according to the following equations:
         phi_y = N_y/ m
         mu_y = 1/ N_y (for each j as y_i = y sum x_i)
         Sigma = 1/len(sample_train _y
          for each j as y_i = y sum (x_i - mu_i)(x_i -mu_i)^T

        :param x: the x train set - domain set
        :param y: the y train set (labels set)
        :return:
        """
        self._trained_X = np.array(x)
        self._trained_y = np.array(y)

        self.x_labeling(0)
        self.x_labeling(1)

        self.find_mu(0)
        self.find_mu(1)

        self.find_phi(0)
        self.find_phi(1)


        self.find_sigma(0)
        self.find_sigma(1)

        self.find_by_tag_eigan_val(0)
        self.find_by_tag_eigan_val(1)

    def find_sigma(self, y_tag):
        """

        :param l:
        :return:
        """
        sigma = 0
        size = self.x_labels[y_tag].shape[1]
        for x in self.x_labels[y_tag]:
            sigma += np.matmul((x - self.mu[y_tag]).reshape(size, 1),
                                 (x - self.mu[y_tag]).reshape(1, size))
        self.sigma_matrix.append(sigma / (len(self.x_labels[y_tag]) - 1))

    def x_labeling(self, i):
        """

        :param i:
        :return:
        """
        self.x_labels.append(self._trained_X[np.where(self._trained_y == i)[0], :])

    def find_mu(self, y_tag):
        """

        :param i:
        :return:
        """
        self.mu.append(
            np.sum(self.x_labels[y_tag], axis=0) / len(self.x_labels[y_tag]))

    def find_phi(self, y_tag):
        """

        :param y_tag:
        :return:
        """
        self.phi.append(len(self.x_labels[y_tag]) / len(self._trained_y))

    def predict(self, x):
        """
        this function predicts the y tag of the x received by the following
        equation:
        h_d(x) = argmax(x^T (Sigma^-1)Mu_y -1/2 (Mu_y)^T(Sigma^-1)(Mu_y)+log(pai_y))
        :param x: the x vector to predict
        :return: its predicted y tag
        """
        h_d = []
        sigma_inverse = []

        # creating Sigma^-1
        for value in self.sigma_matrix:
            sigma_inverse.append(np.linalg.inv(value))

        for tag in range(2):
            #for each y -> h_d(x) =
            # x^T (Sigma^-1)Mu_y -1/2 (Mu_y)^T(Sigma^-1)(Mu_y)+log(pai_y))
            h_d.append(np.dot(np.dot(x, sigma_inverse[tag]),self.mu[tag]) \
                       - 0.5 * (np.dot(np.dot(self.mu[tag],
                                               sigma_inverse[tag]),
                                        self.mu[tag])) + \
                       np.log(self.phi[tag]))

        # argmax - for each y take the maximum
        return np.argmax(h_d)

    def find_eigen_values(self):
        """
        return the egan values of the sigma (covariance) matrix
        :return:
        """
        return self.eigen

    def find_by_tag_eigan_val(self, y_tag):

        self.eigen.append(np.linalg.eigvals(self.sigma_matrix[y_tag]))