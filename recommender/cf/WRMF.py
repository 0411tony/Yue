#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict


class WRMF(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(WRMF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(WRMF, self).initModel()
        self.X=self.P
        self.Y=self.Q

    def buildModel(self):
        userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if not userListen[user].has_key(item[self.recType]):
                    userListen[user][item[self.recType]] = 0
                userListen[user][item[self.recType]] += 1
        print 'training...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            YtY = self.Y.T.dot(self.Y)
            I = np.ones(self.data.getSize(self.recType))
            for user in self.data.name2id['user']:
                C_u = np.ones(self.data.getSize(self.recType))
                P_u = np.zeros(self.data.getSize(self.recType))
                uid = self.data.getId(user,'user')
                for item in userListen[user]:
                    iid = self.data.getId(item,self.recType)
                    r_ui = userListen[user][item]
                    C_u[iid]+=log(1+r_ui/0.01)
                    P_u[iid]=1
                    error = (P_u[iid]-self.X[uid].dot(self.Y[iid]))
                    self.loss+=C_u[iid]*pow(error,2)

                Temp = (YtY+(self.Y.T*(C_u-I)).dot(self.Y)+self.regU*np.eye(self.k))**-1
                self.X[uid] = (Temp.dot(self.Y.T)*C_u).dot(P_u)

            XtX = self.X.T.dot(self.X)
            I = np.ones(self.data.getSize('user'))
            for item in self.data.name2id[self.recType]:
                C_i = np.ones(self.data.getSize('user'))
                P_i = np.zeros(self.data.getSize('user'))
                iid = self.data.getId(item, self.recType)
                for user in self.data.listened[self.recType][item]:
                    uid = self.data.getId(user, 'user')
                    r_ui = self.data.listened[self.recType][item][user]
                    C_i[uid] += log(r_ui/0.01+1)
                    P_i[uid] = 1
                Temp = (XtX+(self.X.T*(C_i-I)).dot(self.X)+self.regU*np.eye(self.k))**-1
                self.Y[iid] = (Temp.dot(self.X.T)*C_i).dot(P_i)

            #self.loss += (self.X * self.X).sum() + (self.Y * self.Y).sum()
            iteration += 1
            print 'iteration:',iteration
            # if self.isConverged(iteration):
            #     break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Y.dot(self.X[u])



