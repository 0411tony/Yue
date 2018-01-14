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
            A = self.Y.T.dot(self.Y)
            B = self.X.T.dot(self.X)
            for user in self.data.name2id['user']:
                C_u = np.ones(self.data.getSize(self.recType))
                P_u = np.zeros(self.data.getSize(self.recType))
                uid = self.data.getId(user,'user')
                for item in self.data.userRecord[user]:
                    iid = self.data.getId(item[self.recType],self.recType)
                    r_ui = self.data.trackListened[item[self.recType]][user]
                    C_u[iid]+=40*r_ui
                    P_u[iid]=1

                LEFT = (A+self.Y.T*(C_u-np.ones(self.data.getSize(self.recType))).dot(self.Y)+self.regU*np.eye(self.k))**-1
                self.P[uid] = LEFT.dot(self.Y.T)*C_u.dot(P_u)




            self.loss = 0

            self.loss += self.regU * (self.X * self.X).sum() + self.regU * (self.Y * self.Y).sum()
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Q.dot(self.P[u])



