
import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        length = len(list(self.labels))
        for i in range(length):
            index = np.where(y == i)
            y_edit = np.zeros(y.shape[0])
            y_edit[index] = 1
            clf = svm.LinearSVC(random_state = 12345)
            model = clf.fit(X,y_edit)
            binary_svm[i] = model
        return binary_svm
        pass

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        length = len(list(self.labels))
        for i in range(length):
            for j in range(length):
                if i !=j and i<j:
                    index_0 = np.where(y == i)
                    index_1 = np.where(y == j)
                    x_edit = np.r_[X[index_0],X[index_1]]
                    y_ones = np.ones(y[index_0].shape[0])
                    y_zeros = np.zeros(y[index_1].shape[0])
                    y_edit = np.r_[y_ones,y_zeros]
                    clf = svm.LinearSVC(random_state = 111)
                    model = clf.fit(x_edit,y_edit)
                    binary_svm[i,j] = model
        return binary_svm
        pass

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        clf = self.binary_svm
        for index in clf:
            scores.append(clf[index].decision_function(X))
        scores = np.array(scores).transpose()
        return scores
        pass

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        length = len(list(self.labels))
        scores = np.zeros((X.shape[0], length))
        clf = self.binary_svm
        for [i,j] in clf:
            pred = clf[i,j].predict(X)
            add_item = np.where(pred == 1)
            del_item = np.where(pred == 0)
            scores[add_item,i] += 1
            scores[del_item,j] += 1
        return scores
        pass

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        loss_1 = 1 / 2 * np.square(np.linalg.norm(W))
        one_array = np.ones((X.shape[0],W.shape[0]))
        eye_array = np.eye(W.shape[0])
        y_array = eye_array[y]
        cal_array = (W.dot(X.transpose()).transpose())
        max_array = one_array - y_array + cal_array
        summary = C * np.max(max_array,axis = 1) - cal_array[np.where(y_array == 1)]
        loss_2 = np.sum(summary)
        loss = loss_1 + loss_2
        return loss
        pass

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        one_array = np.ones((X.shape[0],W.shape[0]))
        eye_array = np.eye(W.shape[0])
        y_array = eye_array[y]
        cal_array = (W.dot(X.transpose()).transpose())
        max_array = one_array - y_array + cal_array
        max_index = np.argmax(max_array,axis = 1)
        list = []
        for i in range(W.shape[0]):
            index_0 = np.where(y == np.unique(y)[i])
            grad_0 = np.sum(X[index_0],axis=0)
            index_1 = np.where (max_index == i)[0]
            if index_1.shape ==(1,0):
                grad_1 = np.zeros((1,X.shape[1]))
            else:
                grad_1 = np.sum(X[index_1],axis=0)
            list.append(grad_1-grad_0)
        grad_array = np.array(list)
        grad = W + C * grad_array
        return grad
        pass
