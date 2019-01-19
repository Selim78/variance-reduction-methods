import numpy as np
import solvers

class LogisticRegression:
    """
    Parameters
    ----------
    l1: float
        Used to specify the LASSO regression penalty
    l2: float
        Used to specify the ridge regression penalty
    solver: str, {'GD', 'SGD', 'SAG', 'SAGA', 'SVRG'}
        Optimization method
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG, one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA, one iteration uses only one data point.

    Attributes
    ----------
    coef_: array
        Coefficient of the features in the decision function.
    """

    SOLVER_SELECTOR = {'GD'         : solvers.GD,
                       'SGD'        : solvers.SGD,
                       'SAG'        : solvers.SAG,
                       'SAGA'       : solvers.SAGA,
                       'SVRG'       : solvers.SVRG,
                       'hogwildSGD' : solvers.hogwildSGD}

    def __init__(self, solver='GD', l1=0.05, l2=0.1, max_iter=100):

        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter

        self.solver = LogisticRegression.SOLVER_SELECTOR.get(solver, None)
        if self.solver == None:
            raise BaseException("Invalid Solver")

    def fit(self, X, y):
        """
        To fit the model to the data (X,y),
        we minimise the empirical risk R_n(w) over w:

            1/n \sum_{i=1}^n(f_i(w)) + r(w),  where  f_i(w) = LogisticLoss(y_i, w^T X_i)
                                                     r(w)   = l1 |w| + 1/2 * l2 ||w||^2

        To minimise this quantity we separate it into:
            1) an L-smooth component:  1/n \sum_{i=1}^n(f_i(w)) + 1/2 * l2 ||w||^2
            2) a non-smooth component: l1 |w|

        We give the *gradient* of 1) and the *proximal* of 2) to a minimiser (GD, SGD, etc.).
        """

        def grad(w, i=None):
            if i is None: # return batch gradient
                return sum(self._loss_grad(X, y, w))/len(y) + self.l2*w
            else: # return an estimate of the gradient computed with the input output (X[i], y[i])
                return self._loss_grad_i(X[i], y[i], w) + self.l2*w

        def prox(w, stepsize):
             return np.vectorize(self._soft_threshold)(w, self.l1 * stepsize)

        # Smoothness constant
        grad.L = 0.25*max(np.linalg.norm(X,2,axis=1))**2 + self.l2
        # Dataset size
        grad.n = len(y)
        # Strong convexity constant
        grad.mu = self.l2
        # Initialisation of the solver
        w0=np.zeros(X.shape[1])

        self.coef_, self._coef_tab = self.solver(w0, grad, prox, self.max_iter)

        self._empirical_risk = lambda w: sum(self._loss(X, y, w))/len(y)     \
                                         + self.l2/2.0 * np.linalg.norm(w,2) \
                                         + self.l1 * np.linalg.norm(w,1)

    def decision_function(self, X):
        """
        Predict confidence scores for the samples in X.
        """
        if not hasattr(self, "coef_"):
            raise BaseException("The model hasn't been fitted yet.")

        return 1.0/(1 + np.exp(-np.dot(X, self.coef_)))

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if not hasattr(self, "coef_"):
            raise BaseException("The model hasn't been fitted yet.")

        return [[-1, 1][x >= 0.5] for x in self.decision_function(X)]

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.
        """
        return np.sum(self.predict(X) == y)/len(y)


    # _____________ private functions _____________

    def _loss(self, X, y, w):
        return np.log(1 + np.exp(-y*np.dot(X, w)))

    def _loss_grad(self, X, y, w):
        return np.diag(-y/(1 + np.exp(y*np.dot(X, w)))) @ X

    def _loss_grad_i(self, Xi, yi, w):
        return -yi/(1 + np.exp(yi*np.dot(Xi, w))) * Xi

    def _soft_threshold(self, x, threshold):
        if x >= threshold:
            return (x - threshold)
        elif x < -threshold:
            return (x + threshold)
        else:
            return 0
