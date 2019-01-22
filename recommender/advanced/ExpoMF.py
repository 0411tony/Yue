from base.IterativeRecommender import IterativeRecommender
from scipy.sparse import *
from scipy import *
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
from math import sqrt

EPS = 1e-8
# this algorithm refers to the following paper:
# #########----  Modeling User Exposure in Recommendation   ----#############

class ExpoMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ExpoMF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(ExpoMF, self).initModel()
        self.lam_theta = 1e-5
        self.lam_beta = 1e-5
        self.lam_y = 1.0
        self.init_mu = 0.01
        self.a = 1.0
        self.b = 99.0
        self.init_std = 0.01
        self.theta = self.init_std * \
            np.random.randn(self.m, self.k).astype(np.float32)
        self.beta = self.init_std * \
            np.random.randn(self.n, self.k).astype(np.float32)
        self.mu = self.init_mu * np.ones(self.n, dtype=np.float32)
        self.n_jobs=1
        self.batch_size=300
        row,col,val = [],[],[]
        for user in self.data.userRecord:
            u = self.data.getId(user, 'user')
            for item in self.data.userRecord[user]:
                i = self.data.getId(item['track'], 'track')
                row.append(u)
                col.append(i)
                val.append(1)
        self.X = csr_matrix((np.array(val),(np.array(row),np.array(col))),(self.m,self.n))

    def buildModel(self):
        print('training...')
        n_users = self.X.shape[0]
        XT = self.X.T.tocsr()  # pre-compute this
        for i in range(self.maxIter):
            print('ITERATION #%d' % i)
            self._update_factors(self.X, XT)
            self._update_expo(self.X, n_users)


    def _update_factors(self, X, XT):
        '''Update user and item collaborative factors with ALS'''
        print('update factors...')
        self.theta = recompute_factors(self.beta, self.theta, X,
                                       self.lam_theta / self.lam_y,
                                       self.lam_y,
                                       self.mu,
                                       self.n_jobs,
                                       batch_size=self.batch_size)

        self.beta = recompute_factors(self.theta, self.beta, XT,
                                      self.lam_beta / self.lam_y,
                                      self.lam_y,
                                      self.mu,
                                      self.n_jobs,
                                      batch_size=self.batch_size)


    def _update_expo(self, X, n_users):
        '''Update exposure prior'''
        print('\tUpdating exposure prior...')

        start_idx = list(range(0, n_users, self.batch_size))
        end_idx = start_idx[1:] + [n_users]

        A_sum = np.zeros_like(self.mu)
        for lo, hi in zip(start_idx, end_idx):
            A_sum += a_row_batch(X[lo:hi], self.theta[lo:hi], self.beta,
                                 self.lam_y, self.mu).sum(axis=0)
        print(self.mu)
        self.mu = (self.a + A_sum - 1) / (self.a + self.b + n_users - 2)


    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        if self.data.contains(u,'user'):
            u = self.data.getId(u,'user')
            return self.beta.dot(self.theta[u])
        else:
            return [self.data.globalMean] * len(self.listened['track'])

# Utility functions #



def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]

def a_row_batch(Y_batch, theta_batch, beta, lam_y, mu):
    '''Compute the posterior of exposure latent variables A by batch'''
    pEX = sqrt(lam_y / 2 * np.pi) * \
          np.exp(-lam_y * theta_batch.dot(beta.T) ** 2 / 2)
    #print pEX.shape,mu.shape
    A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
    A[Y_batch.nonzero()] = 1.
    return A

def _solve(k, A_k, X, Y, f, lam, lam_y, mu):
    '''Update one single factor'''
    s_u, i_u = get_row(Y, k)
    a = np.dot(s_u * A_k[i_u], X[i_u])
    B = X.T.dot(A_k[:, np.newaxis] * X) + lam * np.eye(f)
    return LA.solve(B, a)

def _solve_batch(lo, hi, X, X_old_batch, Y, m, f, lam, lam_y, mu):
    '''Update factors by batch, will eventually call _solve() on each factor to
    keep the parallel process busy'''
    assert X_old_batch.shape[0] == hi - lo

    if mu.size == X.shape[0]:  # update users
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu)
    else:  # update items
        A_batch = a_row_batch(Y[lo:hi], X_old_batch, X, lam_y, mu[lo:hi,
                                                               np.newaxis])

    X_batch = np.empty_like(X_old_batch, dtype=X_old_batch.dtype)
    for ib, k in enumerate(range(lo, hi)):
        X_batch[ib] = _solve(k, A_batch[ib], X, Y, f, lam, lam_y, mu)
    return X_batch

def recompute_factors(X, X_old, Y, lam, lam_y, mu, n_jobs, batch_size=100):
    '''Regress X to Y with exposure matrix (computed on-the-fly with X_old) and
    ridge term lam by embarrassingly parallelization. All the comments below
    are in the view of computing user factors'''
    m, n = Y.shape  # m = number of users, n = number of items
    assert X.shape[0] == n
    assert X_old.shape[0] == m
    f = X.shape[1]  # f = number of factors

    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]

    res = Parallel(n_jobs=n_jobs)(delayed(_solve_batch)(
        lo, hi, X, X_old[lo:hi], Y, m, f, lam, lam_y, mu)
                                  for lo, hi in zip(start_idx, end_idx))

    X_new = np.vstack(res)
    return X_new