from base.IterativeRecommender import IterativeRecommender
import numpy as np
from tool import config
from collections import defaultdict
from math import log,exp
from scipy.sparse import *
from scipy import *

class CoFactor(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(CoFactor, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(CoFactor, self).readConfiguration()
        extraSettings = config.LineConfig(self.config['CoFactor'])
        self.negCount = int(extraSettings['-k']) #the number of negative samples
        if self.negCount < 1:
            self.negCount = 1
        self.regR = float(extraSettings['-gamma'])
        self.filter = int(extraSettings['-filter'])

    def printAlgorConfig(self):
        super(CoFactor, self).printAlgorConfig()
        print('Specified Arguments of', self.config['recommender'] + ':')
        print('k: %d' % self.negCount)
        print('regR: %.5f' %self.regR)
        print('filter: %d' %self.filter)
        print('=' * 80)

    def initModel(self):
        super(CoFactor, self).initModel()
        self.num_items = self.n
        self.num_users = self.m
        #constructing SPPMI matrix
        self.SPPMI = defaultdict(dict)

        self.userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if item[self.recType] not in self.userListen[user]:
                    self.userListen[user][item[self.recType]] = 0
                self.userListen[user][item[self.recType]] += 1

        print('Constructing SPPMI matrix...')
        #for larger data set has many items, the process will be time consuming
        occurrence = defaultdict(dict)
        i=0
        for item1 in self.data.name2id[self.recType]:
            i += 1
            if i % 100 == 0:
                print(str(i) + '/' + str(self.num_items))
            uList1 = self.data.listened[self.recType][item1]

            if len(self.data.trackRecord[item1]) < self.filter:
                continue
            for item2 in self.data.name2id[self.recType]:
                if item1 == item2:
                    continue
                if item2 not in occurrence[item1]:
                    uList2 = self.data.listened[self.recType][item2]
                    if len(self.data.trackRecord[item2]) < self.filter:
                        continue
                    count = len(set(uList1).intersection(set(uList2)))
                    if count > self.filter:
                        occurrence[item1][item2] = count
                        occurrence[item2][item1] = count

        maxVal = 0
        frequency = {}
        for item1 in occurrence:
            frequency[item1] = sum(list(occurrence[item1].values())) * 1.0
        D = sum(list(frequency.values())) * 1.0
        # maxx = -1
        for item1 in occurrence:
            for item2 in occurrence[item1]:
                try:
                    val = max([log(occurrence[item1][item2] * D / (frequency[item1] * frequency[item2])) - log(
                        self.negCount), 0])
                except ValueError:
                    print(self.SPPMI[item1][item2])
                    print(self.SPPMI[item1][item2] * D / (frequency[item1] * frequency[item2]))

                if val > 0:
                    if maxVal < val:
                        maxVal = val
                    self.SPPMI[item1][item2] = val
                    self.SPPMI[item2][item1] = self.SPPMI[item1][item2]
        #normalize
        for item1 in self.SPPMI:
            for item2 in self.SPPMI[item1]:
                self.SPPMI[item1][item2] = self.SPPMI[item1][item2]/maxVal


    def buildModel(self):
        iteration = 0

        self.X=self.P*10 #Theta
        self.Y=self.Q*10 #Beta
        self.w = np.random.rand(self.num_items) / 10  # bias value of item
        self.c = np.random.rand(self.num_items) / 10  # bias value of context
        self.G = np.random.rand(self.num_items, self.k) / 10  # context embedding

        print('training...')
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            YtY = self.Y.T.dot(self.Y)
            for user in self.data.name2id['user']:
                H = np.ones(self.num_items)
                val, pos = [],[]
                P_u = np.zeros(self.num_items)
                uid = self.data.getId(user,'user')
                for item in self.userListen[user]:
                    iid = self.data.getId(item,self.recType)
                    r_ui = self.userListen[user][item]
                    pos.append(iid)
                    val.append(10 * r_ui)
                    H[iid] += 10 * r_ui
                    P_u[iid] = 1
                    error = (P_u[iid] - self.X[uid].dot(self.Y[iid]))
                    self.loss += pow(error, 2)
                # sparse matrix
                C_u = coo_matrix((val, (pos, pos)), shape=(self.num_items, self.num_items))
                A = (YtY + np.dot(self.Y.T, C_u.dot(self.Y)) + self.regU * np.eye(self.k))
                self.X[uid] = np.dot(np.linalg.inv(A), (self.Y.T * H).dot(P_u))

            XtX = self.X.T.dot(self.X)
            for item in self.data.name2id[self.recType]:
                P_i = np.zeros(self.num_users)
                iid = self.data.getId(item, self.recType)
                H = np.ones(self.num_users)
                val,pos = [],[]
                for user in self.data.listened[self.recType][item]:
                    uid = self.data.getId(user, 'user')
                    r_ui = self.data.listened[self.recType][item][user]
                    pos.append(uid)
                    val.append(10*r_ui)
                    H[uid] += 10*r_ui
                    P_i[uid] = 1

                matrix_g1 = np.zeros((self.k,self.k))
                matrix_g2 = np.zeros((self.k,self.k))
                vector_m1 = np.zeros(self.k)
                vector_m2 = np.zeros(self.k)
                update_w = 0
                update_c = 0

                if len(self.SPPMI[item])>0:
                    for context in self.SPPMI[item]:
                        cid = self.data.getId(context, self.recType)
                        gamma = self.G[cid]
                        beta = self.Y[cid]
                        matrix_g1 += gamma.reshape(self.k,1).dot(gamma.reshape(1,self.k))
                        vector_m1 += (self.SPPMI[item][context]-self.w[iid]-self.c[cid])*gamma

                        matrix_g2 += beta.reshape(self.k,1).dot(beta.reshape(1,self.k))
                        vector_m2 += (self.SPPMI[item][context] - self.w[cid]-self.c[iid]) * beta

                        update_w += self.SPPMI[item][context]-self.Y[iid].dot(gamma)-self.c[cid]
                        update_c += self.SPPMI[item][context]-beta.dot(self.G[iid])-self.w[cid]

                C_i = coo_matrix((val, (pos, pos)), shape=(self.num_users, self.num_users))
                A = (XtX + np.dot(self.X.T, C_i.dot(self.X)) + self.regU * np.eye(self.k) + matrix_g1)
                self.Y[iid] = np.dot(np.linalg.inv(A), (self.X.T * H).dot(P_i) + vector_m1)
                if len(self.SPPMI[item]) > 0:
                    self.G[iid] = np.dot(np.linalg.inv(matrix_g2+self.regR * np.eye(self.k)),vector_m2)
                    self.w[iid] = update_w/len(self.SPPMI[item])
                    self.c[iid] = update_c/len(self.SPPMI[item])

            # self.loss += (self.X * self.X).sum() + (self.Y * self.Y).sum()
            iteration += 1
            print('iteration:', iteration, 'loss:', self.loss)
            # if self.isConverged(iteration):
            #     break

    def predictForRanking(self,u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Y.dot(self.X[u])
