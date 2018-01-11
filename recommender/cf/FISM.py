#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool.config import LineConfig
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
class FISM(IterativeRecommender):



    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(FISM, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(FISM, self).initModel()
        self.Bi = np.random.rand(len(self.data.id2name[self.recType]))/100
        self.P = np.random.rand(len(self.data.id2name[self.recType]),self.k)/100

    def readConfiguration(self):
        super(FISM, self).readConfiguration()
        self.rho = int(LineConfig(self.config['FISM'])['-rho'])
        if self.rho<1:
            self.rho=1
        self.alpha = float(LineConfig(self.config['FISM'])['-alpha'])

    def buildModel(self):
        userListened = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                userListened[user][item[self.recType]] = 1

        print 'training...'
        iteration = 0
        itemList = self.data.name2id[self.recType].keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.data.userRecord:
                nu = len(self.data.userRecord[user])
                if nu==1:
                    continue
                coef = pow(nu - 1, -self.alpha)
                sum_Pj = np.zeros(self.k)
                for item in self.data.userRecord[user]:
                    j = self.data.getId(item[self.recType], self.recType)
                    sum_Pj += self.P[j]
                X = []
                for item in self.data.userRecord[user]:
                    x = np.zeros(self.k)
                    i = self.data.getId(item[self.recType],self.recType)
                    for count in range(self.rho):
                        item_j = choice(itemList)
                        while (userListened[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.data.getId(item_j,self.recType)
                        r_ui=coef*(sum_Pj-self.P[i]).dot(self.Q[i])+self.Bi[i]
                        r_uj=coef*(sum_Pj-self.P[j]).dot(self.Q[j])+self.Bi[j]
                        error = 1-(r_ui-r_uj)
                        self.loss += 0.5*error**2
                        #update
                        self.Bi[i]+=self.lRate*(error-self.regB*self.Bi[i])
                        self.Bi[j]-=self.lRate*(error+self.regB*self.Bi[j])
                        self.Q[i]+=self.lRate*(error*coef*(sum_Pj-self.P[i])-self.regI*self.Q[i])
                        self.Q[j]-=self.lRate*(error*coef*(sum_Pj-self.P[j])+self.regI*self.Q[j])
                        x+=error*(self.Q[i]-self.Q[j])
                    X.append(x)

                for ind,item in enumerate(self.data.userRecord[user]):
                    j = self.data.getId(item[self.recType], self.recType)
                    self.P[j]+=self.lRate*(1/float(self.rho)*coef*X[ind]-self.regI*self.P[j])

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum() + self.regB*(self.Bi.dot(self.Bi))
            iteration += 1
            if self.isConverged(iteration):
                break



    def predict(self, user):
        'invoked to rank all the items for the user'
        u = self.data.getId(user,'user')
        #a trick for quick matrix computation
        sum_Pj = np.zeros(self.k)
        for item in self.data.userRecord[user]:
            j = self.data.getId(item[self.recType], self.recType)
            sum_Pj += self.P[j]
        return self.Bi+self.Q.dot(sum_Pj)-(self.P*self.Q).sum(axis=1)



