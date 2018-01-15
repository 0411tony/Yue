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

        print 'training...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.data.name2id['user']:
                C_u = np.ones(self.data.getSize(self.recType))
                P_u = np.zeros(self.data.getSize(self.recType))
                uid = self.data.getId(user,'user')
                for item in self.data.userRecord[user]:
                    iid = self.data.getId(item[self.recType],self.recType)
                    r_ui = self.data.listened[self.recType][item[self.recType]][user]
                    C_u[iid]+=log(1+r_ui)
                    P_u[iid]=1
                    self.loss+=C_u[iid]*(1-self.X[uid].dot(self.Y[iid]))**2

                Temp = ((self.Y.T*C_u).dot(self.Y)+self.regU*np.eye(self.k))**-1
                self.X[uid] = (Temp.dot(self.Y.T)*C_u).dot(P_u)


            for item in self.data.name2id[self.recType]:
                C_i = np.ones(self.data.getSize('user'))
                P_i = np.zeros(self.data.getSize('user'))
                iid = self.data.getId(item, self.recType)
                for user in self.data.listened[self.recType][item]:
                    uid = self.data.getId(user, 'user')
                    r_ui = self.data.listened[self.recType][item][user]
                    C_i[uid] += log(r_ui+1)
                    P_i[uid] = 1
                Temp = ((self.X.T * C_i).dot(self.X) + self.regU * np.eye(self.k)) ** -1
                self.Y[iid] = (Temp.dot(self.X.T) * C_i).dot(P_i)

            self.loss += self.regU * (self.X * self.X).sum() + self.regU * (self.Y * self.Y).sum()
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Y.dot(self.X[u])



