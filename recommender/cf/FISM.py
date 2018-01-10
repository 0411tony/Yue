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
        self.rho = float(LineConfig(self.config['FISM'])['-rho'])
        self.alpha = float(LineConfig(self.config['FISM'])['-alpha'])

    def buildModel(self):        #
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
                u = self.data.getId(user,'user')
                nu = len(self.data.userRecord[user])
                if nu==1:
                    continue
                X = []
                for item in self.data.userRecord[user]:
                    x = np.zeros(self.k)
                    i = self.data.getId(item[self.recType],self.recType)
                    t = self.estimate_t(user, item[self.recType])
                    for count in range(int(self.rho)+1):
                        item_j = choice(itemList)
                        while (userListened[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.data.getId(item_j,self.recType)
                        r_ui=t.dot(self.Q[i])
                        r_uj=t.dot(self.Q[j])
                        error = 1-(r_ui-r_uj)
                        self.loss += 0.5*error**2
                        #update
                        self.Bi[i]+=self.lRate*(error-self.regB*self.Bi[i])
                        self.Bi[j]-=self.lRate*(error+self.regB*self.Bi[j])
                        self.Q[i]+=self.lRate*(error*t-self.regI*self.Q[i])
                        self.Q[j]-=self.lRate*(error*t+self.regI*self.Q[j])
                        x+=error*(self.Q[i]-self.Q[j])
                    X.append(x)
                coef = pow(len(self.data.userRecord[user]) - 1, -self.alpha)
                for ind,item in enumerate(self.data.userRecord[user]):
                    j = self.data.getId(item[self.recType], self.recType)
                    self.P[j]+=self.lRate*(1/float(self.rho)*coef*X[ind]-self.regI*self.P[j])

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum() + self.regB*(self.Bi.dot(self.Bi))
            iteration += 1
            if self.isConverged(iteration):
                break

    def estimate_t(self,user,item):
        t=np.zeros(self.k)
        i = self.data.getId(item,self.recType)
        for ratedItem in self.data.userRecord[user]:
            if ratedItem[self.recType]!=item:
                j = self.data.getId(ratedItem[self.recType],self.recType)
                t += self.P[j]
        t*=pow(len(self.data.userRecord[user])-1,-self.alpha)*t
        t+=self.Bi[i]
        return t



    def predict(self, user):
        'invoked to rank all the items for the user'
        u = self.data.getId(user,'user')
        #a trick for quick matrix computation
        coef = pow(len(self.data.userRecord[user])-1,-self.alpha)
        t = self.estimate_t(user,'')
        return self.Bi+self.Q.dot(t)-(self.P*self.Q).sum(axis=0)*coef



